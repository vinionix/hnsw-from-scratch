from __future__ import annotations

import numpy as np


class HNSWIndex:
    """Implementation contract for the hand-written HNSW index.

    The benchmark suite only depends on build() and search().
    The internal algorithm is intentionally left for the repository owner.
    """

    def __init__(
        self,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
    ) -> None:
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

    def build(self, vectors: np.ndarray) -> None:
        raise NotImplementedError("implement HNSW build/insertion here")

    def search(self, query: np.ndarray, k: int = 10) -> list[int]:
        raise NotImplementedError("implement HNSW search here")
