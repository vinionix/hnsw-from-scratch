from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _parse_value(value: str) -> float | str:
    try:
        return float(value)
    except ValueError:
        return value


def load_rows(path: Path) -> list[dict[str, float | str]]:
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            {key: _parse_value(value) for key, value in row.items()}
            for row in reader
        ]


def _values(rows: list[dict[str, float | str]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def _plot_pair(ax, n: list[float], rows, exact_key: str, hnsw_key: str, title: str, ylabel: str) -> None:
    ax.plot(n, _values(rows, exact_key), marker="o", label="Exact")
    ax.plot(n, _values(rows, hnsw_key), marker="o", label="HNSW")
    ax.set_title(title)
    ax.set_xlabel("Effective searchable embeddings")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()


def save_dashboard(
    rows: list[dict[str, float | str]],
    output: Path,
) -> None:
    n = _values(rows, "n")
    k = int(float(rows[0]["k"]))
    dataset = str(rows[0].get("dataset", "unknown"))

    figure, axes = plt.subplots(2, 4, figsize=(20, 10))

    _plot_pair(
        axes[0, 0],
        n,
        rows,
        "exact_p50_ms",
        "hnsw_p50_ms",
        "Latency p50",
        "Latency (ms)",
    )
    _plot_pair(
        axes[0, 1],
        n,
        rows,
        "exact_p95_ms",
        "hnsw_p95_ms",
        "Latency p95",
        "Latency (ms)",
    )
    _plot_pair(
        axes[0, 2],
        n,
        rows,
        "exact_mean_ms",
        "hnsw_mean_ms",
        "Mean latency",
        "Latency (ms)",
    )

    recall = [value * 100.0 for value in _values(rows, "recall_at_k")]
    axes[0, 3].plot(n, recall, marker="o")
    axes[0, 3].set_title(f"Recall@{k}")
    axes[0, 3].set_xlabel("Effective searchable embeddings")
    axes[0, 3].set_ylabel("Recall (%)")
    axes[0, 3].set_ylim(0.0, 105.0)
    axes[0, 3].grid(True, alpha=0.25)

    _plot_pair(
        axes[1, 0],
        n,
        rows,
        "exact_build_s",
        "hnsw_build_s",
        "Build time",
        "Seconds",
    )
    _plot_pair(
        axes[1, 1],
        n,
        rows,
        "exact_memory_delta_mb",
        "hnsw_memory_delta_mb",
        "Memory delta",
        "RSS delta (MB)",
    )

    axes[1, 2].plot(n, _values(rows, "speedup_p95"), marker="o")
    axes[1, 2].axhline(1.0, linestyle="--", linewidth=1)
    axes[1, 2].set_title("p95 speedup")
    axes[1, 2].set_xlabel("Effective searchable embeddings")
    axes[1, 2].set_ylabel("Exact p95 / HNSW p95")
    axes[1, 2].grid(True, alpha=0.25)

    config_ax = axes[1, 3]
    config_ax.axis("off")
    config_lines = [
        "Benchmark configuration",
        "",
        f"dataset: {dataset}",
        f"dimensions: {int(float(rows[0]['dimensions']))}",
        f"queries: {int(float(rows[0]['queries']))}",
        f"k: {k}",
        f"m: {int(float(rows[0]['m']))}",
        f"ef_construction: {int(float(rows[0]['ef_construction']))}",
        f"ef_search: {int(float(rows[0]['ef_search']))}",
    ]
    if dataset == "clustered":
        config_lines.extend(
            [
                f"clusters: {int(float(rows[0]['clusters']))}",
                f"cluster spread: {float(rows[0]['cluster_spread']):.2f}",
            ]
        )
    config_ax.text(
        0.05,
        0.95,
        "\n".join(config_lines),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
    )

    figure.suptitle(
        f"Exact cosine search vs HNSW — {dataset} dataset",
        fontsize=16,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results",
        nargs="?",
        default="benchmarks/results/results.csv",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    rows = load_rows(results_path)
    output_path = results_path.parent / "benchmark_dashboard.png"

    save_dashboard(rows, output_path)
    print(f"created {output_path}")


if __name__ == "__main__":
    main()
