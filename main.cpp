#include "include/config.h"
#include "include/vector_utils.h"
#include "include/openFHE_wrapper.h"
#include "openfhe.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include "utils.cpp"
#include "include/client.h"
#include "include/server.h"

using namespace lbcrypto;
using namespace std;
using namespace VectorUtils;


namespace {

// function to check the vector length isn't longer than the batch size
// then pad it with 0s since our embeddings are shorter
void padPackedSlots(std::vector<double> &v, size_t batchSize) {
    if (v.size() > batchSize) {
        throw std::runtime_error("padPackedSlots: vector longer than CKKS batch size");
    }
    v.resize(batchSize, 0.0);
}

} // namespace

// TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
int main(int argc, char *argv[]) {

    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <embedding_file> <faiss_file> <db_file>" << std::endl;
        return 1;
    }

    std::string embedding_file = argv[1];
    std::string faiss_file = argv[2];
    std::string database_file = argv[3];

    std::cout << "Embedding file: " << embedding_file << std::endl;
    std::cout << "Faiss file: " << faiss_file << std::endl;
    std::cout << "Database file: " << database_file << std::endl;

    // Query
    std::vector<float> query_embedding = readFloatsFromFile(embedding_file);
    std::vector<std::string> database = readStringsFromFile(database_file);
    faiss::Index *index = readFaissIndex(faiss_file);
    std::vector<std::vector<float>> embedding_database = faissIndexToVectors(index);

    std::cout << "Index loaded successfully!" << std::endl;
    std::cout << "Number of vectors: " << index->ntotal << std::endl;
    std::cout << "Dimension: " << index->d << std::endl;
    std::cout << "Is trained: " << (index->is_trained ? "yes" : "no") << std::endl;

    const size_t dim = query_embedding.size();
    if (dim != static_cast<size_t>(index->d)) {
        std::cerr << "Error: query embedding length (" << dim << ") != index dimension (" << index->d << ")\n";
        return 1;
    }
    for (size_t i = 0; i < embedding_database.size(); ++i) {
        if (embedding_database[i].size() != dim) {
            std::cerr << "Error: database vector " << i << " has length " << embedding_database[i].size()
                      << ", expected " << dim << "\n";
            return 1;
        }
    }

    //Setup Client and Server
    size_t multDepth = OpenFHEWrapper::computeRequiredDepth(5);

    CryptoContext<DCRTPoly> cc;
    cc->ClearEvalMultKeys();
    cc->ClearEvalAutomorphismKeys();
    CryptoContextFactory<DCRTPoly>::ReleaseAllContexts();
    PublicKey<DCRTPoly> pk;
    PrivateKey<DCRTPoly> sk;
    size_t batchSize;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecurityLevel(HEStd_128_classic);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(45);
    parameters.SetScalingTechnique(FIXEDMANUAL);

    cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    batchSize = cc->GetEncodingParams()->GetBatchSize();
    // batch size error check
    if (dim > batchSize) {
        std::cerr << "Error: embedding dimension " << dim << " exceeds CKKS batch size " << batchSize << "\n";
        return 1;
    }

    cout << "Generating key pair... " << endl;
    auto keyPair = cc->KeyGen();
    pk = keyPair.publicKey;
    sk = keyPair.secretKey;

    cout << "Generating mult keys... " << endl;
    cc->EvalMultKeyGen(sk);

    cout << "Generating sum keys... " << endl;
    cc->EvalSumKeyGen(sk);

    cout << "Generating rotation keys... " << endl;
    vector<int> rotationFactors(VECTOR_DIM-1);
    // generate keys from 1 to VECTOR_DIM
    iota(rotationFactors.begin(), rotationFactors.end(), 1);
    // generate positive binary rotation keys greater than VECTOR_DIM
    for(int i = VECTOR_DIM; i < int(batchSize); i *= 2) {
        rotationFactors.push_back(i);
    }
    // generate negative binary rotation keys
    for(int i = 1; i < int(batchSize); i *= 2) {
        rotationFactors.push_back(-i);
    }
    cc->EvalRotateKeyGen(sk, rotationFactors);

    cout << "CKKS scheme set up (depth = " << multDepth << ", batch size = " << batchSize << ")" << endl;

    // Set up database size - 100 for texting 
    const size_t kMaxDbVectors = 50;
    const size_t db_size = (kMaxDbVectors == 0) ? embedding_database.size()
                                                : std::min(kMaxDbVectors, embedding_database.size());
    std::cout << "Configured DB size: " << db_size << std::endl;
    using Clock = std::chrono::steady_clock;
    std::chrono::nanoseconds initDuration(0);

    /// PLAINTEXT APPROACH
    const auto initSquaresStart = Clock::now();
    const float square_query_embedding = square(query_embedding);
    std::vector<float> square_embedding_database(db_size);
    for (size_t i = 0; i < db_size; i++) {
        square_embedding_database[i] = square(embedding_database[i]);
    }
    initDuration += std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now() - initSquaresStart);

    std::vector<float> plaintext_distances(db_size);
    for (size_t i = 0; i < db_size; i++) {
        plaintext_distances[i] = euclideanDistance(
            query_embedding, embedding_database[i], square_query_embedding, square_embedding_database[i]);
    }

    // add to file 
    {
        std::ofstream out("plaintext_distances.txt");
        if (!out.is_open()) {
            std::cerr << "Could not open plaintext_distances.txt\n";
            return 1;
        }
        for (size_t i = 0; i < plaintext_distances.size(); i++) {
            out << i << "," << plaintext_distances[i] << "\n";
        }
    }

    // plaintext thresholding
    // Compare in similarity space for clearer threshold semantics:
    // sim = 1 - (d^2 / 2), then sim >= MATCH_THRESHOLD.
    double kSqDistanceThreshold = 2.0 * (1.0 - MATCH_THRESHOLD);
    if (kSqDistanceThreshold < 0.0) {
        kSqDistanceThreshold = 0.0;
    } else if (kSqDistanceThreshold > 4.0) {
        kSqDistanceThreshold = 4.0;
    }
    std::cout << "Similarity threshold: " << MATCH_THRESHOLD << std::endl;
    std::cout << "Distance threshold: " << kSqDistanceThreshold << std::endl;
    std::vector<float> plaintext_similarity_scores(db_size);
    std::vector<float> plaintext_threshold_bits(db_size);
    for (size_t i = 0; i < db_size; i++) {
        double similarity = 1.0 - static_cast<double>(plaintext_distances[i]) / 2.0;
        if (similarity < -1.0) {
            similarity = -1.0;
        } else if (similarity > 1.0) {
            similarity = 1.0;
        }
        plaintext_similarity_scores[i] = static_cast<float>(similarity);
        plaintext_threshold_bits[i] =
            static_cast<double>(plaintext_similarity_scores[i]) >= MATCH_THRESHOLD ? 1.0f : 0.0f;
    }
    // add to file 
    {
        std::ofstream out("plaintext_thresholds.txt");
        if (!out.is_open()) {
            std::cerr << "Could not open plaintext_thresholds.txt\n";
            return 1;
        }
        for (size_t i = 0; i < plaintext_threshold_bits.size(); i++) {
            out << i << "," << static_cast<int>(plaintext_threshold_bits[i]) << "\n";
        }
    }


    // ENCRYPTED APPROACH
    const auto encryptedTotalStart = Clock::now();
    const auto initEncryptedSetupStart = Clock::now();
    std::vector<double> query_embedding_d(query_embedding.begin(), query_embedding.end());
    padPackedSlots(query_embedding_d, batchSize);
    // Encrypt query embedding
    Plaintext ptE = cc->MakeCKKSPackedPlaintext(query_embedding_d);
    Ciphertext<DCRTPoly> ctE = cc->Encrypt(pk, ptE);

    // Encrypt query embedding squared
    const double e2_plain = static_cast<double>(square_query_embedding);
    Plaintext ptE2Slots = cc->MakeCKKSPackedPlaintext(std::vector<double>(batchSize, e2_plain));

    // Encrypt -2 vector 
    std::vector<float> distances(db_size);
    Plaintext ptMinusTwo = cc->MakeCKKSPackedPlaintext(std::vector<double>(batchSize, -2.0));
    std::vector<Ciphertext<DCRTPoly>> ctDistances(db_size);
    initDuration += std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now() - initEncryptedSetupStart);

    std::chrono::nanoseconds distanceCoreDuration(0);
    for (size_t i = 0; i < db_size; i++) {

        // printing progress for testing
        if (i % 10 == 0) {
            std::cout << "Processing vector " << i << " / " << db_size << std::endl;
        }
        
        // encode database vectors in plaintext
        std::vector<double> d_vec_d(embedding_database[i].begin(), embedding_database[i].end());
        padPackedSlots(d_vec_d, batchSize);
        Plaintext ptD = cc->MakeCKKSPackedPlaintext(d_vec_d);

        // encrypted inner product <e, d>- componentwise multiply
        Ciphertext<DCRTPoly> ctED = cc->EvalMult(ctE, ptD);
        // sum all slots for inner product 
        Ciphertext<DCRTPoly> ctInner = OpenFHEWrapper::sumAllSlots(cc, ctED);

        // -2<e,d>
        const auto distanceCoreStart = Clock::now();
        Ciphertext<DCRTPoly> ctMinus2Inner = cc->EvalMult(ctInner, ptMinusTwo);

        // (||d||^2) as plaintext replicated across slots
        const double d2 = static_cast<double>(square_embedding_database[i]);
        Plaintext ptD2 = cc->MakeCKKSPackedPlaintext(std::vector<double>(batchSize, d2));

        // Distance^2 = ||d||^2 + ||e||^2 - 2<e,d>
        Ciphertext<DCRTPoly> ctDist = cc->EvalAdd(ctMinus2Inner, ptD2);
        ctDist = cc->EvalAdd(ctDist, ptE2Slots);
        distanceCoreDuration += std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - distanceCoreStart);
        ctDistances[i] = ctDist;

        // decrypt distance^2
        Plaintext ptDist;
        cc->Decrypt(sk, ctDist, &ptDist);
        ptDist->SetLength(1);
        const auto vals = ptDist->GetRealPackedValue();
        const double raw = vals.empty() ? 0.0 : static_cast<double>(vals[0]);
        distances[i] = static_cast<float>(std::max(0.0, raw));
    }

    // add to file 
    {
        std::ofstream out("distances.txt");
        if (!out.is_open()) {
            std::cerr << "Could not open distances.txt\n";
            return 1;
        }
        for (size_t i = 0; i < distances.size(); i++) {
            out << distances[i] << "\n";
        }
    }

    // Chebyshev thresholding in similarity space (same decision rule as plaintext):
    // sim = 1 - (d^2 / 2), compare sim against MATCH_THRESHOLD.
    const auto thresholdStart = Clock::now();
    std::vector<Ciphertext<DCRTPoly>> distanceThresholds(db_size);
    for (size_t i = 0; i < db_size; i++) {
        Ciphertext<DCRTPoly> ctSimilarity = cc->EvalMult(ctDistances[i], -0.5);
        cc->EvalAddInPlace(ctSimilarity, 1.0);

        // step ≈ 0 if sim < MATCH_THRESHOLD, else ≈ 2
        Ciphertext<DCRTPoly> ctStep =
            OpenFHEWrapper::chebyshevCompare(cc, ctSimilarity, MATCH_THRESHOLD, COMP_DEPTH);
        distanceThresholds[i] = ctStep;
    }
    const auto thresholdEnd = Clock::now();

    // decrypt the thresholds to read them to check results (soft value ≈ 0 or ≈ 2 before hard cut)
    std::vector<float> distanceThresholdsPT(db_size);
    for (size_t i = 0; i < db_size; i++) {
        Plaintext ptInd;
        cc->Decrypt(sk, distanceThresholds[i], &ptInd);
        ptInd->SetLength(1);

        const auto vals = ptInd->GetRealPackedValue();
        const double v = vals.empty() ? 0.0 : vals[0];
        distanceThresholdsPT[i] = static_cast<float>(v > 1.0 ? 1.0 : 0.0);
    }

    // save encrypted thresholds to file
    {
        std::ofstream out("encrypted_thresholds.txt");
        if (!out.is_open()) {
            std::cerr << "Could not open encrypted_thresholds.txt\n";
            return 1;
        }
        for (size_t i = 0; i < distanceThresholdsPT.size(); i++) {
            out << i << "," << static_cast<int>(distanceThresholdsPT[i]) << "\n";
        }
    }
    const auto encryptedTotalEnd = Clock::now();

    const auto initMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(initDuration).count();
    const auto distanceCalcMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(distanceCoreDuration).count();
    const auto thresholdMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(thresholdEnd - thresholdStart).count();
    const auto encryptedTotalMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(encryptedTotalEnd - encryptedTotalStart).count();

    std::cout << "\nTiming summary (encrypted pipeline)\n";
    std::cout << "  Running total (start -> encrypted output): " << encryptedTotalMs << " ms\n";
    std::cout << "  Initialization (query encrypt + query/db squaring + -2 vector prep): " << initMs << " ms\n";
    std::cout << "  Distance calculation (-2<d,e> + d^2 + e^2, no sq/encrypt/decrypt): " << distanceCalcMs << " ms\n";
    std::cout << "  Thresholding: " << thresholdMs << " ms\n";

    // Compare plaintext vs encrypted threshold bits
    size_t threshold_matches = 0;
    for (size_t i = 0; i < db_size; i++) {
        if (plaintext_threshold_bits[i] == distanceThresholdsPT[i]) {
            threshold_matches++;
        }
    }
    const double threshold_accuracy = (db_size == 0)
                                          ? 0.0
                                          : 100.0 * static_cast<double>(threshold_matches) / static_cast<double>(db_size);
    std::cout << "Threshold agreement: " << threshold_matches << " / " << db_size
              << " (" << threshold_accuracy << "%)" << std::endl;

    std::vector<std::string> result(db_size);
    std::vector<size_t> solutions;
    for (size_t i = 0; i < db_size; i++) {
        if (distanceThresholdsPT[i] == 1.0f && i < database.size()) {
            result[i] = database[i];
            solutions.push_back(i);
        } else {
            result[i] = "0";
        }
    }

    std::cout << "Number of solutions " << solutions.size() << "\n";
    for (size_t id : solutions) {
        std::cout << "  id " << id << "\n";
    }
    for (const std::string &row : result) {
        std::cout << row << "\n";
    }

    const int k = 10;
    std::vector<float> faiss_query(query_embedding.begin(), query_embedding.end());
    std::vector<float> top_distances(k);
    std::vector<faiss::idx_t> labels(k);
    index->search(1, faiss_query.data(), k, top_distances.data(), labels.data());

    std::cout << "\nTop " << k << " nearest neighbors:\n";
    for (int i = 0; i < k; ++i) {
        if (labels[i] >= 0 && static_cast<size_t>(labels[i]) < database.size()) {
            std::cout << "  ID: " << labels[i] << " " << database[labels[i]] << std::endl;
        }
    }

    return 0;
}

// TIP See CLion help at <a
// href="https://www.jetbrains.com/help/clion/">jetbrains.com/help/clion/</a>.
//  Also, you can try interactive lessons for CLion by selecting
//  'Help | Learn IDE Features' from the main menu.

// Timing metric:
// 1) Running total (start -> encrypted output):
//    Starts at encryptedTotalStart and ends at encryptedTotalEnd.
//    Includes encrypted initialization, distance loop, threshold compute,
//    threshold decrypt/hard-cut, and writing encrypted_thresholds.txt.
//
// 2) Initialization (query encrypt + query/db squaring + -2 vector prep):
//    Accumulated initDuration from:
//      - query and database squaring (square_query_embedding, square_embedding_database)
//      - encrypted setup (query pack/encrypt, ptE2Slots, ptMinusTwo, pre-loop allocations)
//
// 3) Distance calculation (-2<d,e> + d^2 + e^2, no sq/encrypt/decrypt):
//    Accumulated distanceCoreDuration inside the per-vector loop from
//    right before EvalMult(ctInner, ptMinusTwo) through the two EvalAdd
//    calls building ctDist
//
// 4) Thresholding:
//    Starts at thresholdStart and ends at thresholdEnd
//    Includes encrypted margin + Chebyshev compare + scaling by 0.5
//    Excludes decrypting threshold ciphertexts and hard thresholding to 0/1

