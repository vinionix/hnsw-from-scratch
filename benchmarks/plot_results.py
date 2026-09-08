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


def save_latency_plot(
    rows: list[dict[str, float | str]],
    output: Path,
) -> None:
    n = [float(row["n"]) for row in rows]
    exact = [float(row["exact_p95_ms"]) for row in rows]
    hnsw = [float(row["hnsw_p95_ms"]) for row in rows]
    dataset = str(rows[0].get("dataset", "unknown"))

    plt.figure()
    plt.plot(n, exact, marker="o", label="Exact p95")
    plt.plot(n, hnsw, marker="o", label="HNSW p95")
    plt.xlabel("Effective searchable embeddings")
    plt.ylabel("Latency (ms)")
    plt.title(f"Exact search vs HNSW ({dataset})")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def save_recall_plot(
    rows: list[dict[str, float | str]],
    output: Path,
) -> None:
    n = [float(row["n"]) for row in rows]
    recall = [float(row["recall_at_k"]) for row in rows]
    k = int(float(rows[0]["k"]))
    dataset = str(rows[0].get("dataset", "unknown"))

    plt.figure()
    plt.plot(n, recall, marker="o")
    plt.xlabel("Effective searchable embeddings")
    plt.ylabel(f"Recall@{k}")
    plt.ylim(0.0, 1.05)
    plt.title(f"HNSW Recall@{k} against exact search ({dataset})")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


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
    output_dir = results_path.parent

    latency_path = output_dir / "latency_p95.png"
    recall_path = output_dir / "recall_at_k.png"

    save_latency_plot(rows, latency_path)
    save_recall_plot(rows, recall_path)

    print(f"created {latency_path}")
    print(f"created {recall_path}")


if __name__ == "__main__":
    main()
