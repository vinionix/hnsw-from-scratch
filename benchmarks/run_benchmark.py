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


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    vectors /= norms
    return vectors


def generate_random_vectors(
    count: int,
    dimensions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    vectors = rng.standard_normal(
        size=(count, dimensions),
        dtype=np.float32,
    )
    return normalize_rows(vectors)


def add_cluster_centers(
    vectors: np.ndarray,
    assignments: np.ndarray,
    centers: np.ndarray,
    cluster_spread: float,
    batch_size: int = 4096,
) -> np.ndarray:
    vectors *= cluster_spread
    for start in range(0, len(vectors), batch_size):
        end = min(start + batch_size, len(vectors))
        vectors[start:end] += centers[assignments[start:end]]
    return normalize_rows(vectors)


def generate_clustered_dataset(
    count: int,
    query_count: int,
    dimensions: int,
    cluster_count: int,
    cluster_spread: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    effective_clusters = min(cluster_count, count)
    centers = generate_random_vectors(effective_clusters, dimensions, rng)

    vector_assignments = rng.integers(0, effective_clusters, size=count)
    vectors = generate_random_vectors(count, dimensions, rng)
    vectors = add_cluster_centers(
        vectors,
        vector_assignments,
        centers,
        cluster_spread,
    )

    query_assignments = rng.integers(0, effective_clusters, size=query_count)
    queries = generate_random_vectors(query_count, dimensions, rng)
    queries = add_cluster_centers(
        queries,
        query_assignments,
        centers,
        cluster_spread,
    )
    return vectors, queries


def generate_dataset(
    dataset: str,
    count: int,
    query_count: int,
    dimensions: int,
    cluster_count: int,
    cluster_spread: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if dataset == "random":
        return (
            generate_random_vectors(count, dimensions, rng),
            generate_random_vectors(query_count, dimensions, rng),
        )
    return generate_clustered_dataset(
        count=count,
        query_count=query_count,
        dimensions=dimensions,
        cluster_count=cluster_count,
        cluster_spread=cluster_spread,
        rng=rng,
    )


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def timed_search(
    index,
    queries: np.ndarray,
    k: int,
) -> tuple[list[list[int]], list[float]]:
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
    dataset: str,
    cluster_count: int,
    cluster_spread: float,
) -> dict[str, float | int | str]:
    rng = np.random.default_rng(seed + size)
    vectors, queries = generate_dataset(
        dataset=dataset,
        count=size,
        query_count=query_count,
        dimensions=dimensions,
        cluster_count=cluster_count,
        cluster_spread=cluster_spread,
        rng=rng,
    )

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
        seed=seed,
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

    speedup_p95 = 0.0
    if hnsw_latency["p95_ms"] > 0.0:
        speedup_p95 = exact_latency["p95_ms"] / hnsw_latency["p95_ms"]

    row: dict[str, float | int | str] = {
        "dataset": dataset,
        "n": size,
        "dimensions": dimensions,
        "queries": query_count,
        "k": k,
        "clusters": cluster_count if dataset == "clustered" else 0,
        "cluster_spread": cluster_spread if dataset == "clustered" else 0.0,
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
        "speedup_p95": speedup_p95,
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
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[1000, 5000, 10000, 25000, 50000],
    )
    parser.add_argument("--dimensions", type=int, default=1536)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--dataset",
        choices=("clustered", "random"),
        default="clustered",
    )
    parser.add_argument("--clusters", type=int, default=32)
    parser.add_argument("--cluster-spread", type=float, default=0.35)
    parser.add_argument("--m", type=int, default=48)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="benchmarks/results/results.csv")
    args = parser.parse_args()

    rows: list[dict] = []
    for size in args.sizes:
        print(
            f"benchmarking dataset={args.dataset} n={size} "
            f"dimensions={args.dimensions} k={args.k}..."
        )
        row = benchmark_size(
            size=size,
            dimensions=args.dimensions,
            query_count=args.queries,
            k=args.k,
            m=args.m,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search,
            seed=args.seed,
            dataset=args.dataset,
            cluster_count=args.clusters,
            cluster_spread=args.cluster_spread,
        )

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
