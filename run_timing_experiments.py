#!/usr/bin/env python3

'''
Timing sweep (default: 20 experiments × 10 vectors = 200 vector evaluations for CSV + plots):

  micromamba run -n naive_rag python run_timing_experiments.py --clean-output

Single 200-vector accuracy check (distances + threshold agreement in project root outputs):

  ./cmake-build-local/naive_rag query_embedding.txt index_1000.faiss index_1000.csv 200
  micromamba run -n naive_rag python compare_thresholds_from_distances.py

Plot after timing:

  micromamba run -n naive_rag python plot_timing_results.py

OpenFHE: if naive_rag fails with exit 127 (EvalChebyshevFunction), rebuild against your
installed OpenFHE or set --openfhe-ld-preload / OPENFHE_LD_PRELOAD to matching .so paths.
Auto-preload of 1.2.4 under /usr/local/lib is applied when present (see --help).
'''

import argparse
import csv
import os
import random
import re
import subprocess
import sys
import shutil
from pathlib import Path

import faiss
import numpy as np


TIMING_PATTERNS = {
    "encrypted_total_ms": re.compile(
        r"Running total (?:\(start -> encrypted output\)|\(encrypted start -> last encrypted output file\)): (\d+) ms"
    ),
    "initialization_ms": re.compile(
        r"(?:Distance initialization \(squaring, query encrypt, ptE2Slots, -2d DB prep\)"
        r"|Initialization \(encrypted: query encrypt \+ ptE2Slots\)"
        r"|Initialization \(query encrypt \+ query/db squaring(?: \+ -2 vector prep|; -2d applied in plaintext per DB vector)\))"
        r": (\d+) ms"
    ),
    "distance_calc_ms": re.compile(
        r"Distance calculation (?:\(-2<d,e> \+ d\^2 \+ e\^2, no sq/encrypt/decrypt\)"
        r"|\(<e,-2d> sum \+ d\^2 \+ e\^2, no decrypt\)"
        r"|\(HE EvalMult\+sumAllSlots\+EvalAdds for d\^2\)): (\d+) ms"
    ),
    "threshold_ms": re.compile(r"Thresholding: (\d+) ms"),
    "threshold_agreement": re.compile(r"Threshold agreement: (\d+) / (\d+) \(([\d.]+)%\)"),
    "solution_count": re.compile(r"Number of solutions (\d+)"),
    "vector_count": re.compile(r"Number of vectors: (\d+)"),
    "vector_dimension": re.compile(r"Dimension: (\d+)"),
    "configured_db_size": re.compile(r"Configured DB size: (\d+)"),
    "distance_threshold": re.compile(
        r"Distance threshold(?: \([^)]+\))?: ([\d.]+)"
    ),
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run naive_rag timing experiments. Defaults: --experiments 20 --set-size 10 "
            "(20 disjoint sets of 10 vectors = 200 vector evaluations per sweep)."
        ),
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
        default=20,
        help="Number of experiments to run.",
    )
    parser.add_argument(
        "--set-size",
        type=int,
        default=10,
        help="Vectors per experiment (e.g. 20×10=200 total evaluations, or --experiments 1 --set-size 200).",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=1000,
        help="Sample from N vectors starting at --pool-start.",
    )
    parser.add_argument(
        "--pool-start",
        type=int,
        default=0,
        help="Start index for sampling pool (inclusive).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed used to form the experiment sets.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete existing experiment_* folders in output dir before running.",
    )
    parser.add_argument(
        "--no-openfhe-usr-local-ld-prepend",
        action="store_true",
        help=(
            "Do not prepend /usr/local/lib to LD_LIBRARY_PATH for naive_rag. "
            "Useful if you rely on conda-provided OpenFHE only."
        ),
    )
    parser.add_argument(
        "--openfhe-ld-preload",
        type=str,
        default=None,
        help=(
            "Colon-separated list for LD_PRELOAD (e.g. three libOPENFHE*.so.1.2.4 paths). "
            "If unset, uses env OPENFHE_LD_PRELOAD when set; else auto-loads 1.2.4 triple under "
            "/usr/local/lib when those files exist (matches naive_rag built against OpenFHE 1.2.x "
            "while libOPENFHEpke.so.1 may symlink to 1.3.x)."
        ),
    )
    parser.add_argument(
        "--no-openfhe-auto-ld-preload",
        action="store_true",
        help="Disable automatic LD_PRELOAD of OpenFHE 1.2.4 libs from /usr/local/lib.",
    )
    return parser.parse_args()


def _default_openfhe_124_preload() -> str | None:
    """OpenFHE 1.2.4 .so paths if all present (ABI matches typical cmake-built naive_rag)."""
    triple = [
        "/usr/local/lib/libOPENFHEpke.so.1.2.4",
        "/usr/local/lib/libOPENFHEcore.so.1.2.4",
        "/usr/local/lib/libOPENFHEbinfhe.so.1.2.4",
    ]
    if all(Path(p).is_file() for p in triple):
        return ":".join(triple)
    return None


def subprocess_env_for_naive_rag(
    base: dict[str, str],
    *,
    prepend_usr_local_openfhe: bool,
    openfhe_ld_preload: str | None,
    no_openfhe_auto_ld_preload: bool,
) -> dict[str, str]:
    env = dict(base)

    preload = openfhe_ld_preload
    if preload is None:
        preload = env.get("OPENFHE_LD_PRELOAD", "").strip() or None
    if preload is None and not no_openfhe_auto_ld_preload:
        preload = _default_openfhe_124_preload()
    if preload:
        existing = env.get("LD_PRELOAD", "").strip()
        env["LD_PRELOAD"] = f"{preload}:{existing}" if existing else preload

    # Prepend /usr/local/lib only when not using LD_PRELOAD for OpenFHE: mixing
    # LD_LIBRARY_PATH (often resolving libOPENFHEpke.so.1 -> 1.3.x) with 1.2.4 preload
    # can crash (mixed ABI). With preload, conda/micromamba env paths still apply for faiss.
    if prepend_usr_local_openfhe and not preload:
        usrlocal = Path("/usr/local/lib")
        pke = usrlocal / "libOPENFHEpke.so.1"
        if usrlocal.is_dir() and pke.exists():
            prefix = str(usrlocal.resolve())
            prev = env.get("LD_LIBRARY_PATH", "").strip()
            env["LD_LIBRARY_PATH"] = f"{prefix}:{prev}" if prev else prefix

    return env


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


def require_metrics(metrics: dict[str, float | int], experiment_id: int, log_path: Path) -> None:
    required = (
        "encrypted_total_ms",
        "initialization_ms",
        "distance_calc_ms",
        "threshold_ms",
        "vector_count",
        "vector_dimension",
        "configured_db_size",
        "distance_threshold",
        "threshold_agreement_matches",
        "threshold_agreement_total",
        "threshold_accuracy_percent",
    )
    missing = [key for key in required if key not in metrics]
    if missing:
        raise RuntimeError(
            f"Experiment {experiment_id} missing metrics {missing}. "
            f"Check log format in {log_path}."
        )


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
    if args.clean_output:
        for stale_dir in output_dir.glob("experiment_*"):
            if stale_dir.is_dir():
                shutil.rmtree(stale_dir)

    source_index = faiss.read_index(str(index_path))
    database_lines = read_database_lines(database_path)
    max_pool_size = min(source_index.ntotal, len(database_lines))
    if args.pool_start < 0:
        raise ValueError("--pool-start must be >= 0")
    if args.pool_start >= max_pool_size:
        raise ValueError(
            f"--pool-start ({args.pool_start}) exceeds available vectors ({max_pool_size})."
        )
    available_pool = min(args.pool_size, max_pool_size - args.pool_start)
    requested_total = args.experiments * args.set_size
    if requested_total > available_pool:
        raise ValueError(
            f"Need {requested_total} vectors for disjoint experiments, but only {available_pool} are available "
            f"in range [{args.pool_start}, {args.pool_start + available_pool})."
        )

    shuffled_indices = list(range(args.pool_start, args.pool_start + available_pool))
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
            str(args.set_size),
        ]
        completed = subprocess.run(
            command,
            cwd=run_dir,
            text=True,
            capture_output=True,
            env=subprocess_env_for_naive_rag(
                os.environ.copy(),
                prepend_usr_local_openfhe=not args.no_openfhe_usr_local_ld_prepend,
                openfhe_ld_preload=args.openfhe_ld_preload,
                no_openfhe_auto_ld_preload=args.no_openfhe_auto_ld_preload,
            ),
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

        metrics = parse_metrics(log_path.read_text(encoding="utf-8"))
        require_metrics(metrics, experiment_id, log_path)
        row = {
            "experiment_id": experiment_id,
            "selected_indices": ",".join(str(idx) for idx in selected_indices),
            "vector_count": metrics.get("vector_count", args.set_size),
            "configured_db_size": metrics.get("configured_db_size", args.set_size),
            "vector_dimension": metrics.get("vector_dimension", ""),
            "distance_threshold": metrics.get("distance_threshold", 0.60),
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
    print(f"Pool range used: [{args.pool_start}, {args.pool_start + available_pool})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
