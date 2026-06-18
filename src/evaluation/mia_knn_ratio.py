"""k-NN distance-ratio membership inference baseline."""
from __future__ import annotations
import numpy as np
from sklearn.neighbors import NearestNeighbors
class KNNDistanceRatioMembershipInference:
    """Membership inference via log-ratio of mean k-NN distances to member vs. non-member banks.
    score(x) = log((d_non_member + eps) / (d_member + eps)); higher means more member-like.
    """
    def __init__(self, k: int = 5, eps: float = 1e-12) -> None:
        if k <= 0:
            msg = "k must be positive."
            raise ValueError(msg)
        if eps <= 0.0:
            msg = "eps must be positive."
            raise ValueError(msg)
        self.k = int(k)
        self.eps = float(eps)
        self._member_nn: NearestNeighbors | None = None
        self._non_member_nn: NearestNeighbors | None = None
        self._member_k = self.k
        self._non_member_k = self.k
        self._dim: int | None = None
    @staticmethod
    def _as_2d_float(name: str, embeddings: np.ndarray) -> np.ndarray:
        x = np.asarray(embeddings, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
            msg = f"{name} must be a non-empty 1D or 2D embedding array."
            raise ValueError(msg)
        if not np.isfinite(x).all():
            msg = f"{name} contains NaN or infinite values."
            raise ValueError(msg)
        return np.ascontiguousarray(x, dtype=np.float64)
    def fit(
        self,
        shadow_member_embeddings: np.ndarray,
        shadow_non_member_embeddings: np.ndarray,
    ) -> "KNNDistanceRatioMembershipInference":
        """Fit member and non-member k-NN reference banks."""
        members = self._as_2d_float("shadow_member_embeddings", shadow_member_embeddings)
        non_members = self._as_2d_float(
            "shadow_non_member_embeddings", shadow_non_member_embeddings
        )
        if members.shape[1] != non_members.shape[1]:
            msg = "Shadow member and non-member embeddings must have the same dimension."
            raise ValueError(msg)
        self._member_k = min(self.k, members.shape[0])
        self._non_member_k = min(self.k, non_members.shape[0])
        self._dim = int(members.shape[1])
        self._member_nn = NearestNeighbors(n_neighbors=self._member_k, metric="euclidean")
        self._member_nn.fit(members)
        self._non_member_nn = NearestNeighbors(
            n_neighbors=self._non_member_k, metric="euclidean"
        )
        self._non_member_nn.fit(non_members)
        return self
    def score_samples(self, target_embeddings: np.ndarray) -> np.ndarray:
        """Return member-likelihood scores; higher is more member-like."""
        if self._member_nn is None or self._non_member_nn is None or self._dim is None:
            msg = "Call fit() before score_samples()."
            raise RuntimeError(msg)
        x = self._as_2d_float("target_embeddings", target_embeddings)
        if x.shape[1] != self._dim:
            msg = f"Expected embedding dimension {self._dim}, got {x.shape[1]}."
            raise ValueError(msg)
        d_member, _ = self._member_nn.kneighbors(x, n_neighbors=self._member_k)
        d_non_member, _ = self._non_member_nn.kneighbors(x, n_neighbors=self._non_member_k)
        mean_member = d_member.mean(axis=1)
        mean_non_member = d_non_member.mean(axis=1)
        return np.log((mean_non_member + self.eps) / (mean_member + self.eps))
    def evaluate_tpr_at_fpr(
        self,
        target_embeddings: np.ndarray,
        target_labels: np.ndarray,
        fpr_threshold: float = 0.001,
    ) -> float:
        """Return TPR at empirical FPR threshold on target non-members."""
        if not 0.0 < fpr_threshold < 1.0:
            msg = "fpr_threshold must be in (0, 1)."
            raise ValueError(msg)
        labels = np.asarray(target_labels, dtype=np.int64).ravel()
        scores = self.score_samples(target_embeddings)
        if scores.shape[0] != labels.shape[0]:
            msg = "target_embeddings and target_labels must align in row count."
            raise ValueError(msg)
        non_member_scores = scores[labels == 0]
        member_scores = scores[labels == 1]
        if non_member_scores.size == 0 or member_scores.size == 0:
            return 0.0
        tau = float(np.quantile(non_member_scores, 1.0 - fpr_threshold))
        return float(np.mean(member_scores > tau))
