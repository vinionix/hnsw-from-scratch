from __future__ import annotations

import numpy as np


def recall_at_k(expected: list[int], actual: list[int], k: int) -> float:
    expected_set = set(expected[:k])
    if not expected_set:
        return 1.0
    return len(expected_set.intersection(actual[:k])) / len(expected_set)


def latency_summary(samples_ms: list[float]) -> dict[str, float]:
    values = np.asarray(samples_ms, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "mean_ms": float(values.mean()),
    }
