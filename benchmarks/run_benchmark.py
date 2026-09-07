from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.metrics import latency_summary, recall_at_k
from src.brute_force import BruteForceIndex
from src.hnsw import HNSWIndex


def generate_vectors(
    count: int,
    dimensions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    vectors = rng.normal(size=(count, dimensions)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def timed_search(index, queries: np.ndarray, k: int) -> tuple[list[list[int]], list[float]]:
    results: list[list[int]] = []
    latencies: list[float] = []

    for query in queries:
        start = time.perf_counter_ns()
        result = index.search(query, k=k)
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        results.append(result)
        latencies.append(elapsed_ms)

    return results, latencies


def benchmark_size(
    size: int,
    dimensions: int,
    query_count: int,
    k: int,
    m: int,
    ef_construction: int,
    ef_search: int,
    seed: int,
) -> dict[str, float | int | str]:
    rng = np.random.default_rng(seed + size)
    vectors = generate_vectors(size, dimensions, rng)
    queries = generate_vectors(query_count, dimensions, rng)

    exact = BruteForceIndex()
    before_exact_memory = rss_mb()
    start = time.perf_counter()
    exact.build(vectors)
    exact_build_s = time.perf_counter() - start
    exact_memory_mb = max(0.0, rss_mb() - before_exact_memory)

    exact_results, exact_latencies = timed_search(exact, queries, k)
    exact_latency = latency_summary(exact_latencies)

    hnsw = HNSWIndex(
        m=m,
        ef_construction=ef_construction,
        ef_search=ef_search,
    )

    before_hnsw_memory = rss_mb()
    start = time.perf_counter()
    hnsw.build(vectors)
    hnsw_build_s = time.perf_counter() - start
    hnsw_memory_mb = max(0.0, rss_mb() - before_hnsw_memory)

    hnsw_results, hnsw_latencies = timed_search(hnsw, queries, k)
    hnsw_latency = latency_summary(hnsw_latencies)

    recalls = [
        recall_at_k(expected, actual, k)
        for expected, actual in zip(exact_results, hnsw_results, strict=True)
    ]

    row: dict[str, float | int | str] = {
        "n": size,
        "dimensions": dimensions,
        "queries": query_count,
        "k": k,
        "m": m,
        "ef_construction": ef_construction,
        "ef_search": ef_search,
        "exact_build_s": exact_build_s,
        "exact_p50_ms": exact_latency["p50_ms"],
        "exact_p95_ms": exact_latency["p95_ms"],
        "exact_mean_ms": exact_latency["mean_ms"],
        "exact_memory_delta_mb": exact_memory_mb,
        "hnsw_build_s": hnsw_build_s,
        "hnsw_p50_ms": hnsw_latency["p50_ms"],
        "hnsw_p95_ms": hnsw_latency["p95_ms"],
        "hnsw_mean_ms": hnsw_latency["mean_ms"],
        "hnsw_memory_delta_mb": hnsw_memory_mb,
        "recall_at_k": float(np.mean(recalls)),
        "speedup_p95": exact_latency["p95_ms"] / hnsw_latency["p95_ms"],
    }

    del hnsw, exact, vectors, queries
    gc.collect()
    return row


def write_csv(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 5000, 10000, 25000, 50000])
    parser.add_argument("--dimensions", type=int, default=1536)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="benchmarks/results/results.csv")
    args = parser.parse_args()

    rows: list[dict] = []
    for size in args.sizes:
        print(f"benchmarking n={size} dimensions={args.dimensions}...")
        try:
            row = benchmark_size(
                size=size,
                dimensions=args.dimensions,
                query_count=args.queries,
                k=args.k,
                m=args.m,
                ef_construction=args.ef_construction,
                ef_search=args.ef_search,
                seed=args.seed,
            )
        except NotImplementedError as error:
            raise SystemExit(
                "HNSWIndex is still a skeleton. Implement build() and search() in src/hnsw.py first."
            ) from error

        rows.append(row)
        print(
            f"  exact p95={row['exact_p95_ms']:.3f} ms | "
            f"hnsw p95={row['hnsw_p95_ms']:.3f} ms | "
            f"recall@{args.k}={row['recall_at_k']:.3f} | "
            f"speedup={row['speedup_p95']:.2f}x"
        )

    output = Path(args.output)
    write_csv(rows, output)
    print(f"results written to {output}")


if __name__ == "__main__":
    main()
