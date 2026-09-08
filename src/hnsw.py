from __future__ import annotations

import heapq
import math

import numpy as np


class HNSWIndex:
    """Small HNSW implementation for the benchmark spike.

    The index uses cosine distance. Vectors are normalized during build(),
    so cosine similarity becomes a dot product and distance is 1 - similarity.
    """

    def __init__(
        self,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        seed: int = 42,
    ) -> None:
        if m < 2:
            raise ValueError("m must be at least 2")
        if ef_construction < m:
            raise ValueError("ef_construction must be >= m")
        if ef_search < 1:
            raise ValueError("ef_search must be at least 1")

        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        self._rng = np.random.default_rng(seed)
        self._level_multiplier = 1.0 / math.log(m)

        self._vectors: np.ndarray | None = None
        self._links: list[list[list[int]]] = []
        self._levels: list[int] = []
        self._entry_point: int | None = None
        self._max_level = -1

    def build(self, vectors: np.ndarray) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("vectors must be a 2D matrix")

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self._vectors = matrix / norms

        self._links = []
        self._levels = []
        self._entry_point = None
        self._max_level = -1

        for node_id in range(len(self._vectors)):
            self._insert(node_id)

    def search(self, query: np.ndarray, k: int = 10) -> list[int]:
        if self._vectors is None:
            raise RuntimeError("index has not been built")
        if self._entry_point is None or k <= 0:
            return []

        vector = self._normalize_query(query)
        current = self._entry_point

        for level in range(self._max_level, 0, -1):
            current = self._greedy_search(vector, current, level)

        ef = max(self.ef_search, k)
        candidates = self._search_layer(vector, [current], ef, 0)
        return [node_id for _, node_id in candidates[:k]]

    def _insert(self, node_id: int) -> None:
        level = self._random_level()
        self._levels.append(level)
        self._links.append([[] for _ in range(level + 1)])

        if self._entry_point is None:
            self._entry_point = node_id
            self._max_level = level
            return

        assert self._vectors is not None
        vector = self._vectors[node_id]
        current = self._entry_point

        for search_level in range(self._max_level, level, -1):
            current = self._greedy_search(vector, current, search_level)

        highest_shared_level = min(level, self._max_level)
        for search_level in range(highest_shared_level, -1, -1):
            candidates = self._search_layer(
                vector,
                [current],
                self.ef_construction,
                search_level,
            )
            selected = self._select_neighbors(candidates)

            self._links[node_id][search_level].extend(selected)
            for neighbor in selected:
                self._connect_neighbor(neighbor, node_id, search_level)

            if candidates:
                current = candidates[0][1]

        if level > self._max_level:
            self._entry_point = node_id
            self._max_level = level

    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: list[int],
        ef: int,
        level: int,
    ) -> list[tuple[float, int]]:
        candidate_heap: list[tuple[float, int]] = []
        result_heap: list[tuple[float, int]] = []
        visited = set(entry_points)

        for node_id in entry_points:
            distance = self._distance(query, node_id)
            heapq.heappush(candidate_heap, (distance, node_id))
            heapq.heappush(result_heap, (-distance, node_id))

        while candidate_heap:
            current_distance, current = heapq.heappop(candidate_heap)
            worst_distance = -result_heap[0][0]

            if len(result_heap) >= ef and current_distance > worst_distance:
                break

            neighbors = self._neighbors_at_level(current, level)
            unseen = [node for node in neighbors if node not in visited]
            if not unseen:
                continue

            visited.update(unseen)
            distances = self._distances(query, unseen)

            for neighbor, distance in zip(unseen, distances):
                distance = float(distance)
                worst_distance = -result_heap[0][0]

                if len(result_heap) < ef or distance < worst_distance:
                    heapq.heappush(candidate_heap, (distance, neighbor))
                    heapq.heappush(result_heap, (-distance, neighbor))

                    if len(result_heap) > ef:
                        heapq.heappop(result_heap)

        return sorted(
            [
                (-negative_distance, node_id)
                for negative_distance, node_id in result_heap
            ]
        )

    def _greedy_search(
        self,
        query: np.ndarray,
        entry_point: int,
        level: int,
    ) -> int:
        current = entry_point
        current_distance = self._distance(query, current)

        while True:
            neighbors = self._neighbors_at_level(current, level)
            if not neighbors:
                return current

            distances = self._distances(query, neighbors)
            best_position = int(np.argmin(distances))
            best_distance = float(distances[best_position])

            if best_distance >= current_distance:
                return current

            current = neighbors[best_position]
            current_distance = best_distance

    def _select_neighbors(
        self,
        candidates: list[tuple[float, int]],
    ) -> list[int]:
        return [node_id for _, node_id in candidates[: self.m]]

    def _connect_neighbor(
        self,
        node_id: int,
        new_neighbor: int,
        level: int,
    ) -> None:
        links = self._links[node_id][level]
        if new_neighbor not in links:
            links.append(new_neighbor)

        limit = self._max_connections(level)
        if len(links) <= limit:
            return

        assert self._vectors is not None
        node_vector = self._vectors[node_id]
        neighbor_matrix = self._vectors[links]
        distances = 1.0 - (neighbor_matrix @ node_vector)
        keep_positions = np.argpartition(distances, limit - 1)[:limit]

        self._links[node_id][level] = [
            links[int(position)] for position in keep_positions
        ]

    def _neighbors_at_level(self, node_id: int, level: int) -> list[int]:
        if level > self._levels[node_id]:
            return []
        return self._links[node_id][level]

    def _max_connections(self, level: int) -> int:
        if level == 0:
            return self.m * 2
        return self.m

    def _random_level(self) -> int:
        sample = max(float(self._rng.random()), np.finfo(float).tiny)
        return int(-math.log(sample) * self._level_multiplier)

    def _normalize_query(self, query: np.ndarray) -> np.ndarray:
        assert self._vectors is not None
        vector = np.asarray(query, dtype=np.float32)

        if vector.ndim != 1:
            raise ValueError("query must be a 1D vector")
        if vector.shape[0] != self._vectors.shape[1]:
            raise ValueError("query dimension does not match index")

        norm = np.linalg.norm(vector)
        if norm != 0.0:
            vector = vector / norm
        return vector

    def _distance(self, query: np.ndarray, node_id: int) -> float:
        assert self._vectors is not None
        return float(1.0 - np.dot(self._vectors[node_id], query))

    def _distances(
        self,
        query: np.ndarray,
        node_ids: list[int],
    ) -> np.ndarray:
        assert self._vectors is not None
        return 1.0 - (self._vectors[node_ids] @ query)
