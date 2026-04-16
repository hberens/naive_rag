#!/usr/bin/env python3

'''LD_PRELOAD=/usr/local/lib/libOPENFHEpke.so.1.2.3:/usr/local/lib/libOPENFHEcore.so.1.2.3:/usr/local/lib/libOPENFHEbinfhe.so.1.2.3 \
~/.local/bin/micromamba run -n naive_rag \
python run_timing_experiments.py \
  --experiments 5 \
  --set-size 10 \
  --index ./index_1000.faiss \
  --database ./index_1000.csv \
  --query ./query_embedding.txt \
  --output-dir ./timing_experiments'''

import argparse
import csv
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import faiss
import numpy as np


TIMING_PATTERNS = {
    "encrypted_total_ms": re.compile(r"Running total \(start -> encrypted output\): (\d+) ms"),
    "initialization_ms": re.compile(
        r"Initialization \(query encrypt \+ query/db squaring \+ -2 vector prep\): (\d+) ms"
    ),
    "distance_calc_ms": re.compile(
        r"Distance calculation \(-2<d,e> \+ d\^2 \+ e\^2, no sq/encrypt/decrypt\): (\d+) ms"
    ),
    "threshold_ms": re.compile(r"Thresholding: (\d+) ms"),
    "threshold_agreement": re.compile(r"Threshold agreement: (\d+) / (\d+) \(([\d.]+)%\)"),
    "solution_count": re.compile(r"Number of solutions (\d+)"),
    "vector_count": re.compile(r"Number of vectors: (\d+)"),
    "vector_dimension": re.compile(r"Dimension: (\d+)"),
    "configured_db_size": re.compile(r"Configured DB size: (\d+)"),
    "distance_threshold": re.compile(r"Distance threshold: ([\d.]+)"),
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated 10-vector timing experiments for naive_rag."
    )
    parser.add_argument(
        "--binary",
        default="./cmake-build-local/naive_rag",
        help="Path to the compiled naive_rag binary.",
    )
    parser.add_argument(
        "--query",
        default="./query_embedding.txt",
        help="Path to the query embedding file.",
    )
    parser.add_argument(
        "--index",
        default="./index_1000.faiss",
        help="Path to the source FAISS index.",
    )
    parser.add_argument(
        "--database",
        default="./index_1000.csv",
        help="Path to the source database text file.",
    )
    parser.add_argument(
        "--output-dir",
        default="./timing_experiments",
        help="Directory where experiment outputs will be written.",
    )
    parser.add_argument(
        "--experiments",
        type=int,
        default=10,
        help="Number of experiments to run.",
    )
    parser.add_argument(
        "--set-size",
        type=int,
        default=10,
        help="Number of vectors per experiment.",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=1000,
        help="Only sample from the first N vectors of the source index/database.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed used to form the experiment sets.",
    )
    return parser.parse_args()


def read_database_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def build_subset_index(source_index: faiss.Index, indices: list[int]) -> faiss.Index:
    vectors = []
    for idx in indices:
        vector = source_index.reconstruct(int(idx))
        vectors.append(vector)

    subset_index = faiss.IndexFlatL2(source_index.d)
    subset_index.add(np.asarray(vectors, dtype="float32"))
    return subset_index


def parse_metrics(stdout: str) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    for key, pattern in TIMING_PATTERNS.items():
        match = pattern.search(stdout)
        if not match:
            continue
        if key == "threshold_agreement":
            metrics["threshold_agreement_matches"] = int(match.group(1))
            metrics["threshold_agreement_total"] = int(match.group(2))
            metrics["threshold_accuracy_percent"] = float(match.group(3))
        elif key == "distance_threshold":
            metrics[key] = float(match.group(1))
        else:
            metrics[key] = int(match.group(1))
    return metrics


def main() -> int:
    args = parse_args()

    binary_path = Path(args.binary).resolve()
    query_path = Path(args.query).resolve()
    index_path = Path(args.index).resolve()
    database_path = Path(args.database).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not binary_path.exists():
        raise FileNotFoundError(f"Binary not found: {binary_path}")
    if not query_path.exists():
        raise FileNotFoundError(f"Query file not found: {query_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")
    if not database_path.exists():
        raise FileNotFoundError(f"Database file not found: {database_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    source_index = faiss.read_index(str(index_path))
    database_lines = read_database_lines(database_path)
    available_pool = min(args.pool_size, source_index.ntotal, len(database_lines))
    requested_total = args.experiments * args.set_size
    if requested_total > available_pool:
        raise ValueError(
            f"Need {requested_total} vectors for disjoint experiments, but only {available_pool} are available."
        )

    shuffled_indices = list(range(available_pool))
    random.Random(args.seed).shuffle(shuffled_indices)

    results: list[dict[str, object]] = []
    for experiment_number in range(args.experiments):
        start = experiment_number * args.set_size
        end = start + args.set_size
        selected_indices = sorted(shuffled_indices[start:end])
        experiment_id = experiment_number + 1
        run_dir = output_dir / f"experiment_{experiment_id:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        subset_index_path = run_dir / "subset.faiss"
        subset_database_path = run_dir / "subset.csv"
        log_path = run_dir / "run.log"

        subset_index = build_subset_index(source_index, selected_indices)
        faiss.write_index(subset_index, str(subset_index_path))
        subset_database_path.write_text(
            "\n".join(database_lines[idx] for idx in selected_indices) + "\n",
            encoding="utf-8",
        )

        command = [
            str(binary_path),
            str(query_path),
            str(subset_index_path),
            str(subset_database_path),
        ]
        completed = subprocess.run(
            command,
            cwd=run_dir,
            text=True,
            capture_output=True,
            env=os.environ.copy(),
            check=False,
        )

        combined_output = completed.stdout
        if completed.stderr:
            combined_output += "\n[stderr]\n" + completed.stderr
        log_path.write_text(combined_output, encoding="utf-8")

        if completed.returncode != 0:
            raise RuntimeError(
                f"Experiment {experiment_id} failed with exit code {completed.returncode}. "
                f"See {log_path}."
            )

        metrics = parse_metrics(completed.stdout)
        row = {
            "experiment_id": experiment_id,
            "selected_indices": ",".join(str(idx) for idx in selected_indices),
            "vector_count": metrics.get("vector_count", args.set_size),
            "configured_db_size": metrics.get("configured_db_size", args.set_size),
            "vector_dimension": metrics.get("vector_dimension", ""),
            "distance_threshold": metrics.get("distance_threshold", 0.61),
            "encrypted_total_ms": metrics.get("encrypted_total_ms", ""),
            "initialization_ms": metrics.get("initialization_ms", ""),
            "distance_calc_ms": metrics.get("distance_calc_ms", ""),
            "threshold_ms": metrics.get("threshold_ms", ""),
            "threshold_agreement_matches": metrics.get("threshold_agreement_matches", ""),
            "threshold_agreement_total": metrics.get("threshold_agreement_total", ""),
            "threshold_accuracy_percent": metrics.get("threshold_accuracy_percent", ""),
            "solution_count": metrics.get("solution_count", ""),
            "log_file": str(log_path.relative_to(output_dir)),
            "subset_index_file": str(subset_index_path.relative_to(output_dir)),
            "subset_database_file": str(subset_database_path.relative_to(output_dir)),
        }
        results.append(row)

    csv_path = output_dir / "timing_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
