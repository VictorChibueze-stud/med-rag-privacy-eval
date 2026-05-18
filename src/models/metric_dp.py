"""Metric DP via Mahalanobis-scaled analytical Gaussian noise.

This module implements the Sprint 2 mechanism described in the project plan. For
an embedding ``x``, the mechanism estimates a local covariance matrix from the
``k`` nearest clean corpus embeddings and draws correlated Gaussian noise whose
geometry follows that neighbourhood covariance. This is a practical local
Mahalanobis approximation to metric DP for sentence embeddings.
"""

from __future__ import annotations

import numpy as np


class MetricDPMechanism:
    """Apply Metric DP noise scaled to local Mahalanobis geometry.

    For each target embedding, the mechanism finds the nearest neighbours in a
    clean reference corpus, estimates their local covariance, and adds a Gaussian
    draw scaled by the analytical Gaussian scalar used elsewhere in this project.

    The implementation avoids a dense ``d × d`` Cholesky factorization for every
    row. For a sample covariance
    ``cov = centered.T @ centered / (k - 1) + ridge * I``, a draw from
    ``N(0, cov)`` can be generated exactly as::

        centered.T @ z_k / sqrt(k - 1) + sqrt(ridge) * z_d

    where ``z_k`` and ``z_d`` are independent standard normal vectors. This keeps
    the mechanism feasible for repeated experiment runs.

    Attributes:
        epsilon: Privacy budget.
        delta: Privacy parameter delta.
        k: Number of neighbours for local covariance estimation.
        ridge: Diagonal covariance stabilizer.
    """

    def __init__(
        self,
        epsilon: float,
        delta: float = 1e-5,
        k: int = 50,
        ridge: float = 1e-6,
    ) -> None:
        """Store privacy and neighbourhood parameters.

        Args:
            epsilon: Positive privacy budget.
            delta: Privacy parameter in ``(0, 1)``.
            k: Positive number of nearest neighbours for covariance estimation.
            ridge: Non-negative diagonal stabilizer added to each local covariance.
        """
        if epsilon <= 0.0 or delta <= 0.0 or delta >= 1.0:
            msg = "Need epsilon > 0 and 0 < delta < 1."
            raise ValueError(msg)
        if k <= 0:
            msg = "k must be positive."
            raise ValueError(msg)
        if ridge < 0.0:
            msg = "ridge must be non-negative."
            raise ValueError(msg)
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.k = int(k)
        self.ridge = float(ridge)

    @property
    def sigma_scalar(self) -> float:
        """Analytical Gaussian scalar with L2 sensitivity constant 2.0."""
        return float(
            2.0 * np.sqrt(2.0 * np.log(1.25 / self.delta)) / self.epsilon
        )

    @staticmethod
    def _as_2d_float(name: str, arr: np.ndarray) -> np.ndarray:
        """Convert an embedding array to a 2-D float64 matrix."""
        x = np.asarray(arr, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            msg = f"{name} must be a 1-D embedding or 2-D embedding matrix."
            raise ValueError(msg)
        return x

    def _nearest_neighbour_indices(
        self, x: np.ndarray, corpus: np.ndarray
    ) -> np.ndarray:
        """Return indices of the ``k`` closest corpus rows to a target vector."""
        diffs = corpus - x
        dists = np.einsum("ij,ij->i", diffs, diffs)
        k_use = min(self.k, corpus.shape[0])
        if k_use == corpus.shape[0]:
            return np.argsort(dists)
        idx = np.argpartition(dists, kth=k_use - 1)[:k_use]
        return idx[np.argsort(dists[idx])]

    def apply_noise(
        self,
        embeddings: np.ndarray,
        corpus_embeddings: np.ndarray,
    ) -> np.ndarray:
        """Add Mahalanobis-calibrated Gaussian noise to embeddings.

        Args:
            embeddings: ``(n, d)`` target embeddings to privatise, or one ``(d,)`` row.
            corpus_embeddings: ``(N, d)`` clean reference corpus for local covariance
                estimation, typically all clean embeddings.

        Returns:
            Noised embeddings with the same shape as ``embeddings`` after 1-D inputs
            are promoted to ``(1, d)``.
        """
        targets = self._as_2d_float("embeddings", embeddings)
        corpus = self._as_2d_float("corpus_embeddings", corpus_embeddings)
        if corpus.shape[0] == 0:
            msg = "corpus_embeddings must contain at least one row."
            raise ValueError(msg)
        if targets.shape[1] != corpus.shape[1]:
            msg = (
                "embeddings and corpus_embeddings must have the same feature "
                f"dimension, got {targets.shape[1]} and {corpus.shape[1]}."
            )
            raise ValueError(msg)

        sigma = self.sigma_scalar
        result = np.empty_like(targets)
        dim = targets.shape[1]

        for i, x_i in enumerate(targets):
            idx = self._nearest_neighbour_indices(x_i, corpus)
            neighbours = corpus[idx]
            mean = np.mean(neighbours, axis=0, keepdims=True)
            centered = neighbours - mean

            if neighbours.shape[0] >= 2:
                # Exact draw from centered.T @ centered / (k - 1) without forming or
                # factorizing the dense covariance matrix.
                z_local = np.random.normal(size=neighbours.shape[0])
                local_noise = centered.T @ z_local / np.sqrt(neighbours.shape[0] - 1)
            else:
                local_noise = np.zeros(dim, dtype=np.float64)

            if self.ridge > 0.0:
                ridge_noise = np.sqrt(self.ridge) * np.random.normal(size=dim)
            else:
                ridge_noise = np.zeros(dim, dtype=np.float64)

            result[i] = x_i + sigma * (local_noise + ridge_noise)

        return result
