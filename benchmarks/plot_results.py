from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            {key: float(value) for key, value in row.items()}
            for row in reader
        ]


def save_latency_plot(rows: list[dict[str, float]], output: Path) -> None:
    n = [row["n"] for row in rows]
    exact = [row["exact_p95_ms"] for row in rows]
    hnsw = [row["hnsw_p95_ms"] for row in rows]

    plt.figure()
    plt.plot(n, exact, marker="o", label="Exact p95")
    plt.plot(n, hnsw, marker="o", label="HNSW p95")
    plt.xlabel("Number of embeddings")
    plt.ylabel("Latency (ms)")
    plt.title("Exact search vs HNSW")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def save_recall_plot(rows: list[dict[str, float]], output: Path) -> None:
    n = [row["n"] for row in rows]
    recall = [row["recall_at_k"] for row in rows]

    plt.figure()
    plt.plot(n, recall, marker="o")
    plt.xlabel("Number of embeddings")
    plt.ylabel("Recall@K")
    plt.ylim(0.0, 1.05)
    plt.title("HNSW recall against exact search")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="benchmarks/results/results.csv")
    args = parser.parse_args()

    results_path = Path(args.results)
    rows = load_rows(results_path)
    output_dir = results_path.parent

    latency_path = output_dir / "latency_p95.png"
    recall_path = output_dir / "recall_at_k.png"

    save_latency_plot(rows, latency_path)
    save_recall_plot(rows, recall_path)

    print(f"created {latency_path}")
    print(f"created {recall_path}")


if __name__ == "__main__":
    main()
