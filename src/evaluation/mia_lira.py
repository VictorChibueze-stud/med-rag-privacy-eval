"""LiRA-style membership inference with multiple shadow models and global variance.

Implements the offline LiRA variant described in Carlini et al. (2022)
"Membership Inference Attacks From First Principles". The attack trains
``n_shadow_models`` independent logistic regression classifiers on bootstrap
resamples of the shadow data, then estimates the out-distribution moments
(mu_out, sigma_out) globally across all shadow models, following the
offline-with-global-variance prescription that achieves within 20% of the
full 64-model attack using as few as 16 models.
"""

import numpy as np
import scipy.stats
from sklearn.linear_model import LogisticRegression


class LiRAMembershipInference:
    """Multi-shadow-model LiRA with bootstrap resampling and global variance.

    ``n_shadow_models`` independent ``LogisticRegression`` classifiers are each
    trained on a bootstrap resample of the shadow member and non-member sets.
    The out-distribution moments ``(mu_out, sigma_out)`` are estimated globally
    by pooling logit scores from all shadow models on the full (non-resampled)
    shadow non-member set. This global variance estimation is the key mechanism
    that makes the offline LiRA variant viable with fewer than 64 models
    (Carlini et al. 2022).

    At evaluation time, per-point scores are averaged across all shadow models
    before the logit transform, yielding a more stable likelihood estimate.

    ``p = 1 - Phi(z)`` is the (upper-tail) p-value: small ``p`` means a large
    *positive* ``z`` = ``(phi - mu_out) / sigma_out`` (more "member-like" than the
    shadow non-member cloud).
    """

    def __init__(self, n_shadow_models: int = 16) -> None:
        """Build the shadow model ensemble; call ``train_shadow_models`` before eval.

        Args:
            n_shadow_models: Number of independent shadow classifiers to train.
                Carlini et al. (2022) show 16 models with global variance estimation
                achieves within 20% of the full 64-model attack.
        """
        self.n_shadow_models = n_shadow_models
        self.shadow_models: list[LogisticRegression] = [
            LogisticRegression(max_iter=1000) for _ in range(n_shadow_models)
        ]
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
        """Train all shadow models on bootstrap resamples; fit global null moments.

        Each of the ``n_shadow_models`` classifiers is trained on an independent
        bootstrap resample (sampling with replacement) of the shadow member and
        non-member sets. After training, logit scores from every shadow model are
        evaluated on the full (non-resampled) shadow non-member set and pooled into
        a single flat array. ``mu_out`` and ``sigma_out`` are the mean and standard
        deviation of this pooled array, providing the globally estimated null moments
        used in the offline LiRA variant (Carlini et al. 2022).

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
        if x_m.shape[0] < self.n_shadow_models:
            msg = (
                f"shadow_member_embeddings has {x_m.shape[0]} rows but "
                f"n_shadow_models={self.n_shadow_models}. Need at least "
                f"n_shadow_models rows to avoid degenerate bootstrap resamples."
            )
            raise ValueError(msg)

        rng = np.random.default_rng(seed=None)  # Uses global numpy seed for repro.

        for model in self.shadow_models:
            # Independent bootstrap resample of each class for this shadow model.
            idx_m = rng.integers(0, x_m.shape[0], size=x_m.shape[0])
            idx_n = rng.integers(0, x_n.shape[0], size=x_n.shape[0])
            x_boot = np.vstack([x_m[idx_m], x_n[idx_n]])
            y_boot = np.concatenate(
                [
                    np.ones(len(idx_m), dtype=np.int64),
                    np.zeros(len(idx_n), dtype=np.int64),
                ]
            )
            model.fit(x_boot, y_boot)

        # Global variance estimation: pool logit scores from all shadow models
        # evaluated on the full (non-resampled) non-member set (Carlini et al. 2022).
        all_phi: list[np.ndarray] = []
        for model in self.shadow_models:
            probs = model.predict_proba(x_n)[:, 1]
            all_phi.append(self._logit(probs))
        pooled = np.concatenate(all_phi)
        self.mu_out = float(np.mean(pooled))
        self.sigma_out = float(np.std(pooled, ddof=0))
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

        Per-point membership scores are averaged across all ``n_shadow_models``
        before the logit transform, yielding a more stable estimate than a single
        model. The logit is then Z-scored using the globally estimated null moments
        ``(mu_out, sigma_out)`` and converted to a one-sided p-value
        ``p = 1 - Phi(z)``. On *target* non-members only, the ``fpr_threshold``
        quantile of those p-values gives threshold ``tau``; TPR is the fraction of
        members with ``p < tau``.

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
        # Average predicted probability across all shadow models for a stable score.
        avg_probs = np.mean(
            np.stack([m.predict_proba(x)[:, 1] for m in self.shadow_models], axis=0),
            axis=0,
        )
        phi_p = self._logit(avg_probs)
        # Standardize w.r.t. globally estimated shadow non-member null (Carlini 2022).
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
