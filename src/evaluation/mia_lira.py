"""LiRA-style membership inference with a shadow logistic and logit / tail scores."""

import numpy as np
import scipy.stats
from sklearn.linear_model import LogisticRegression


class LiRAMembershipInference:
    """Shadow linear classifier, logit calibration, and TPR@FPR on a target set.

    A shadow ``LogisticRegression`` maps 384-d embeddings to ``P(in | x)`` on clean
    shadow data. The logit of that probability is then Z-scored using the (normal)
    moments estimated from *shadow* non-member logits, so the target split can be
    converted to a standard-normal-like scale for tail-based decisions.

    ``p = 1 - Phi(z)`` is the (upper-tail) p-value: small ``p`` means a large
    *positive* ``z`` = ``(phi - mu_out) / sigma_out`` (more "member-like" than the
    shadow non-member cloud).
    """

    def __init__(self) -> None:
        """Build the shadow model; call ``train_shadow_models`` before evaluation."""
        self.shadow_model: LogisticRegression = LogisticRegression(max_iter=1000)
        self.mu_out: float = 0.0
        self.sigma_out: float = 1.0

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        """Map probabilities to the real line with a clip to stay off ±∞."""
        eps = 1e-9
        pc = np.clip(p, eps, 1.0 - eps)
        return np.log(pc / (1.0 - pc))

    def train_shadow_models(
        self,
        shadow_member_embeddings: np.ndarray,
        shadow_non_member_embeddings: np.ndarray,
    ) -> None:
        """Fit the shadow linear model and the non-member calibrator ``(mu, sigma)``.

        After fitting ``P(member | x)`` on concatenated (member, non-member) shadow
        rows, we take **only** the shadow non-member rows, transform predicted
        probabilities with the *logit* to avoid 0/1 edge collapse, and record the
        empirical mean/standard deviation of those logits. Under a Gaussian
        working model (LiRA), ``mu_out`` and ``sigma_out`` act as the null moments
        for the non-member class.

        Args:
            shadow_member_embeddings: ``(n_m, d)`` array of in-shadow member vectors.
            shadow_non_member_embeddings: ``(n_n, d)`` out-of-train (shadow) non-member
                vectors, same feature dimension.
        """
        x_m = np.asarray(shadow_member_embeddings, dtype=np.float64)
        x_n = np.asarray(shadow_non_member_embeddings, dtype=np.float64)
        if x_m.ndim != 2 or x_n.ndim != 2 or x_m.shape[1] != x_n.shape[1]:
            msg = "Member and non-member shadow matrices must be 2-D with the same d."
            raise ValueError(msg)
        x = np.vstack([x_m, x_n])
        y = np.concatenate(
            [
                np.ones(x_m.shape[0], dtype=np.int64),
                np.zeros(x_n.shape[0], dtype=np.int64),
            ]
        )
        self.shadow_model.fit(x, y)
        probs = self.shadow_model.predict_proba(x_n)[:, 1]
        # Logit to stabilize variance near 0/1; maps (0,1) -> R for a linear score.
        phi_p = self._logit(probs)
        self.mu_out = float(np.mean(phi_p))
        self.sigma_out = float(np.std(phi_p, ddof=0))
        if self.sigma_out < 1e-8:
            # Near-degenerate case (e.g. a single point): keep Z-scores finite.
            self.sigma_out = 1e-8

    def evaluate_tpr_at_fpr(
        self,
        target_embeddings: np.ndarray,
        target_labels: np.ndarray,
        fpr_threshold: float = 0.001,
    ) -> float:
        """TPR of "member" calls at a fixed empirical FPR on the target **non**-members.

        We compute the same logit and Z-score pipeline as in training, then
        ``p = 1 - CDF(z)``. On *target* non-members only, the ``fpr_threshold``
        *quantile* of those p-values gives a single cutoff ``tau``; the TPR is the
        fraction of **members** for which ``p < tau`` (i.e. very small tail mass
        under the shadow null), matching a small-FPR "alarm" for membership.

        Args:
            target_embeddings: ``(N, d)`` rows, possibly DP-noised at test time.
            target_labels: ``(N,)`` with ``1`` = member, ``0`` = non-member.
            fpr_threshold: Target false positive rate in ``(0, 1)`` (e.g. ``0.001``).

        Returns:
            The true positive rate in ``[0, 1]`` at the chosen empirical cutoff.
        """
        if not 0.0 < fpr_threshold < 1.0:
            msg = "fpr_threshold must be in (0, 1)."
            raise ValueError(msg)
        x = np.asarray(target_embeddings, dtype=np.float64)
        labels = np.asarray(target_labels, dtype=np.int64).ravel()
        if x.ndim != 2 or x.shape[0] != labels.shape[0]:
            msg = "target_embeddings and target_labels must align in row count."
            raise ValueError(msg)
        p_hat = self.shadow_model.predict_proba(x)[:, 1]
        phi_p = self._logit(p_hat)
        # Standardize w.r.t. shadow *non-member* logit cloud (Z ~ approx N(0,1)).
        z_scores = (phi_p - self.mu_out) / self.sigma_out
        # One-sided tail prob under a Gaussian null on ``z`` (larger z => smaller p).
        p_values = 1.0 - scipy.stats.norm.cdf(z_scores)
        nm = p_values[labels == 0]
        mem = p_values[labels == 1]
        if nm.size == 0 or mem.size == 0:
            return 0.0
        # FPR control on the held-out *target* non-members: lower ``p`` = stronger.
        tau = float(np.quantile(nm, fpr_threshold))
        return float(np.mean(mem < tau))
