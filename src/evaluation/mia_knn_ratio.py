"""k-NN distance-ratio membership inference baseline.

This is intentionally simpler than LiRA: the attacker keeps two auxiliary banks of
shadow embeddings, one member-like and one non-member-like.  A target point is
called more member-like when it is closer to the member bank than to the
non-member bank.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


class KNNDistanceRatioMembershipInference:
    """Membership inference using a k-nearest-neighbour distance ratio.

    For each target embedding ``x`` we compute the mean Euclidean distance to its
    ``k`` nearest shadow-member embeddings (``d_member``) and to its ``k`` nearest
    shadow-non-member embeddings (``d_non_member``).  The member score is

    ``score(x) = log((d_non_member + eps) / (d_member + eps))``.

    Larger scores mean ``x`` is relatively closer to the shadow-member bank and
    therefore more likely to be a member.  This gives a lightweight second MIA
    baseline that can be evaluated with the same TPR-at-FPR protocol as LiRA.
    """

    def __init__(self, k: int = 5, eps: float = 1e-12) -> None:
        """Create an unfitted k-NN distance-ratio attack.

        Args:
            k: Number of nearest neighbours averaged in each reference bank.
            eps: Small constant to keep the distance ratio finite.
        """
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
        """Validate an embedding array and return a contiguous float64 matrix."""
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
        """Fit the two auxiliary k-NN reference banks.

        Args:
            shadow_member_embeddings: ``(n_m, d)`` member-like shadow vectors.
            shadow_non_member_embeddings: ``(n_nm, d)`` non-member-like shadow vectors.

        Returns:
            ``self`` for chaining.
        """
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
        """Return member-likelihood scores for target embeddings.

        Larger scores are more member-like.  Call ``fit`` before this method.
        """
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
        """Evaluate TPR at a fixed empirical FPR on target non-members.

        Args:
            target_embeddings: ``(N, d)`` rows to score.
            target_labels: ``(N,)`` with ``1`` = member and ``0`` = non-member.
            fpr_threshold: Desired false-positive rate in ``(0, 1)``.

        Returns:
            True-positive rate in ``[0, 1]`` for the k-NN ratio attack.
        """
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

        # Higher score = more member-like. Choose the upper-tail non-member cutoff.
        tau = float(np.quantile(non_member_scores, 1.0 - fpr_threshold))
        return float(np.mean(member_scores > tau))
