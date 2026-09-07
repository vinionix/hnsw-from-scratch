from __future__ import annotations

import numpy as np


class BruteForceIndex:
    """Exact cosine-similarity baseline used as ground truth."""

    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None

    def build(self, vectors: np.ndarray) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self._vectors = matrix / norms

    def search(self, query: np.ndarray, k: int = 10) -> list[int]:
        if self._vectors is None:
            raise RuntimeError("index has not been built")

        vector = np.asarray(query, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm != 0.0:
            vector = vector / norm

        scores = self._vectors @ vector
        k = min(k, len(scores))
        top = np.argpartition(scores, -k)[-k:]
        ordered = top[np.argsort(scores[top])[::-1]]
        return ordered.tolist()
