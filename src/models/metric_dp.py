from __future__ import annotations
import numpy as np

class MetricDPMechanism:
    def __init__(self, epsilon: float, delta: float=1e-05, k: int=50, ridge: float=1e-06) -> None:
        if epsilon <= 0.0 or delta <= 0.0 or delta >= 1.0:
            msg = 'Need epsilon > 0 and 0 < delta < 1.'
            raise ValueError(msg)
        if k <= 0:
            msg = 'k must be positive.'
            raise ValueError(msg)
        if ridge < 0.0:
            msg = 'ridge must be non-negative.'
            raise ValueError(msg)
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.k = int(k)
        self.ridge = float(ridge)

    @property
    def sigma_scalar(self) -> float:
        return float(2.0 * np.sqrt(2.0 * np.log(1.25 / self.delta)) / self.epsilon)

    @staticmethod
    def _as_2d_float(name: str, arr: np.ndarray) -> np.ndarray:
        x = np.asarray(arr, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            msg = f'{name} must be 1-D or 2-D.'
            raise ValueError(msg)
        return x

    def _nearest_neighbour_indices(self, x: np.ndarray, corpus: np.ndarray) -> np.ndarray:
        diffs = corpus - x
        dists = np.einsum('ij,ij->i', diffs, diffs)
        k_use = min(self.k, corpus.shape[0])
        if k_use == corpus.shape[0]:
            return np.argsort(dists)
        idx = np.argpartition(dists, kth=k_use - 1)[:k_use]
        return idx[np.argsort(dists[idx])]

    def apply_noise(self, embeddings: np.ndarray, corpus_embeddings: np.ndarray) -> np.ndarray:
        targets = self._as_2d_float('embeddings', embeddings)
        corpus = self._as_2d_float('corpus_embeddings', corpus_embeddings)
        if corpus.shape[0] == 0:
            msg = 'corpus_embeddings must contain at least one row.'
            raise ValueError(msg)
        if targets.shape[1] != corpus.shape[1]:
            msg = f'Dimension mismatch: embeddings {targets.shape[1]} vs corpus {corpus.shape[1]}.'
            raise ValueError(msg)
        sigma = self.sigma_scalar
        result = np.empty_like(targets)
        dim = targets.shape[1]
        for i, x_i in enumerate(targets):
            idx = self._nearest_neighbour_indices(x_i, corpus)
            neighbours = corpus[idx]
            centered = neighbours - np.mean(neighbours, axis=0, keepdims=True)
            if neighbours.shape[0] >= 2:
                z_local = np.random.normal(size=neighbours.shape[0])
                local_noise = centered.T @ z_local / np.sqrt(neighbours.shape[0] - 1)
            else:
                local_noise = np.zeros(dim, dtype=np.float64)
            ridge_noise = np.sqrt(self.ridge) * np.random.normal(size=dim) if self.ridge > 0.0 else np.zeros(dim, dtype=np.float64)
            result[i] = x_i + sigma * (local_noise + ridge_noise)
        return result
