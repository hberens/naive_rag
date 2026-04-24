#!/usr/bin/env python3
"""
Plot timing metrics from timing_experiments/timing_results.csv.

Expected CSV columns (written by run_timing_experiments.py):
- experiment_id
- encrypted_total_ms
- initialization_ms
- distance_calc_ms
- threshold_ms
- vector_count
- vector_dimension
- configured_db_size
- distance_threshold
- selected_indices
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Row:
    experiment_id: int
    encrypted_total_ms: float
    initialization_ms: float
    distance_calc_ms: float
    threshold_ms: float
    vector_count: int
    vector_dimension: int
    configured_db_size: int
    distance_threshold: float
    selected_indices: str


def _as_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _as_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return default


def load_results_csv(path: Path) -> list[Row]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    rows: list[Row] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            rows.append(
                Row(
                    experiment_id=_as_int(raw.get("experiment_id")),
                    encrypted_total_ms=_as_float(raw.get("encrypted_total_ms")),
                    initialization_ms=_as_float(raw.get("initialization_ms")),
                    distance_calc_ms=_as_float(raw.get("distance_calc_ms")),
                    threshold_ms=_as_float(raw.get("threshold_ms")),
                    vector_count=_as_int(raw.get("vector_count")),
                    vector_dimension=_as_int(raw.get("vector_dimension")),
                    configured_db_size=_as_int(raw.get("configured_db_size")),
                    distance_threshold=_as_float(raw.get("distance_threshold")),
                    selected_indices=str(raw.get("selected_indices") or ""),
                )
            )

    rows.sort(key=lambda r: r.experiment_id)
    return rows


def plot_metric_subplots(
    rows: list[Row],
    *,
    output_path: Path,
        title: str = "naive_rag timing metrics by experiment (20 × 10 vectors)",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install it in your env, e.g.\n"
            "  micromamba install -n naive_rag -c conda-forge matplotlib\n"
        ) from exc

    exp_ids = [r.experiment_id for r in rows]
    metrics = [
        ("Encrypted Total (ms)", [r.encrypted_total_ms for r in rows], "#4c78a8"),
        ("Distance init (ms)", [r.initialization_ms for r in rows], "#f58518"),
        ("Distance Calc (ms)", [r.distance_calc_ms for r in rows], "#54a24b"),
        ("Threshold (ms)", [r.threshold_ms for r in rows], "#e45756"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes_flat = axes.flatten()

    for ax, (metric_name, values, color) in zip(axes_flat, metrics, strict=True):
        ax.plot(exp_ids, values, color=color, linewidth=1.4, alpha=0.8)
        ax.scatter(exp_ids, values, color=color, s=40, zorder=3)
        ax.set_title(metric_name)
        ax.set_ylabel("Time (ms)")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_xticks(exp_ids)
        ax.set_xlim(min(exp_ids) - 0.3, max(exp_ids) + 0.3)

    for ax in axes[1]:
        ax.set_xlabel("Experiment ID")

    vec_dim = rows[0].vector_dimension if rows else 0
    vec_count = rows[0].vector_count if rows else 0
    cfg_db = rows[0].configured_db_size if rows else 0
    thresh = rows[0].distance_threshold if rows else 0.0
    fig.suptitle(
        f"{title}\nvector_dimension={vec_dim}, vector_count={vec_count}, configured_db_size={cfg_db}, threshold={thresh}",
        fontsize=12,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot naive_rag timing results.")
    p.add_argument(
        "--csv",
        default="./timing_experiments/timing_results.csv",
        help="Path to timing_results.csv.",
    )
    p.add_argument(
        "--out",
        default="./timing_experiments/timing_results.png",
        help="Output image path (png).",
    )
    p.add_argument(
        "--title",
        default="naive_rag timing metrics by experiment (20 × 10 vectors)",
        help="Figure title.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_results_csv(Path(args.csv).resolve())
    plot_metric_subplots(
        rows, output_path=Path(args.out).resolve(), title=args.title
    )
    print(f"Wrote {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

