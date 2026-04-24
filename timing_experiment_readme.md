# Timing Experiments (default: 20 sets × 10 vectors)

## What these tests do

- Sample disjoint subsets from the first 1000 vectors in `index_1000.faiss` and `index_1000.csv`
- Run `naive_rag` once per subset
- Extract and save the 4 timing metrics:
  - `encrypted_total_ms`
  - `initialization_ms`
  - `distance_calc_ms`
  - `threshold_ms`
- Save additional metadata:
  - `vector_count`
  - `vector_dimension` (expected 768)
  - `configured_db_size`
  - `distance_threshold`
  - `selected_indices` for each experiment

Default experiment layout:
- **20 experiments**
- **10 vectors per experiment**
- Drawn from the first **1000 vectors**

## What each timing metric actually measures

These definitions come from how timing is instrumented in `main.cpp`.

- `encrypted_total_ms`
  - Wall clock from `encryptedTotalStart` (start of encrypted section, after plaintext) to `encryptedTotalEnd`.
  - Includes encrypted query setup, HE distance loop **including** per-vector distance decrypts (for `distances.txt`), writing `distances.txt`, homomorphic thresholding, threshold ciphertext decrypt, hard cut, and writing `encrypted_thresholds.txt`.
  - **Excludes** all plaintext distance/threshold work before `encryptedTotalStart`.

- `initialization_ms` (log line: **Distance initialization …**)
  - **Squaring** query and every DB vector (`square_query_embedding`, `square_embedding_database`).
  - **Encrypt query** (`ctE`) and **ptE2Slots** (`||e||^2` replicated in plaintext slots).
  - **Pack `-2*d`** for every database vector (double vectors padded to batch size), before the HE distance loop.
  - **Does not** include the homomorphic inner-product / `sumAllSlots` work (see `distance_calc_ms`).

- `distance_calc_ms`
  - Per-vector HE only: packed `EvalMult` + `sumAllSlots` (inner) plus plaintext `||d||^2` slot and two `EvalAdd` (tail) to form `ctDist`.
  - **Excludes** decrypting distance ciphertexts (those sit in the encrypted wall-clock / running total, not this bucket).

- `threshold_ms`
  - Homomorphic thresholding only: from `thresholdStart` to `thresholdEnd` in `main.cpp`.
  - Per vector: similarity `sim = 1 - d^2/2` (encrypted mult + add) then `chebyshevCompare`.
  - Stops **before** any `Decrypt` on threshold ciphertexts and before the soft-value → 0/1 cut.

## Files

- `run_timing_experiments.py`: runs experiments and writes outputs
- `plot_timing_results.py`: builds a single graph image from CSV results

## Prerequisites

- Compiled binary at `./cmake-build-local/naive_rag`
- Input files:
  - `./query_embedding.txt`
  - `./index_1000.faiss`
  - `./index_1000.csv`
- Python env `naive_rag` with `faiss` and `numpy`
- `matplotlib` for plotting


## Run the 10x10 timing experiments

From `openfhe/naive_rag`:

```bash
LD_PRELOAD=/usr/local/lib/libOPENFHEpke.so.1.2.3:/usr/local/lib/libOPENFHEcore.so.1.2.3:/usr/local/lib/libOPENFHEbinfhe.so.1.2.3 \
~/.local/bin/micromamba run -n naive_rag \
python run_timing_experiments.py \
  --experiments 10 \
  --set-size 10 \
  --index ./index_1000.faiss \
  --database ./index_1000.csv \
  --query ./query_embedding.txt \
  --output-dir ./timing_experiments
```

## Outputs

In `./timing_experiments/`:

- `timing_results.csv` (best for plotting)
- `timing_results.png` (graph generated from CSV)
- Per-run folders:
  - `experiment_01/`, `experiment_02/`, ... `experiment_10/`
  - each contains `subset.faiss`, `subset.csv`, and `run.log`

## Make the graph

```bash
~/.local/bin/micromamba run -n naive_rag python plot_timing_results.py \
  --csv ./timing_experiments/timing_results.csv \
  --out ./timing_experiments/timing_results.png
```

This creates:
- `./timing_experiments/timing_results.png`

## What the graph shows

The plot contains 4 subplots (2x2), one per timing metric:

- Encrypted Total (ms)
- Initialization (ms)
- Distance Calc (ms)
- Threshold (ms)
