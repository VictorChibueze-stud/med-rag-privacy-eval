"""LiRA membership inference with bootstrap shadow models (Carlini et al. 2022)."""

import numpy as np
import scipy.stats
from sklearn.linear_model import LogisticRegression


class LiRAMembershipInference:
    """Offline LiRA with global variance estimation over n_shadow_models classifiers."""

    def __init__(self, n_shadow_models: int = 16) -> None:
        self.n_shadow_models = n_shadow_models
        self.shadow_models: list[LogisticRegression] = [
            LogisticRegression(max_iter=1000) for _ in range(n_shadow_models)
        ]
        self.mu_out: float = 0.0
        self.sigma_out: float = 1.0

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        eps = 1e-9
        pc = np.clip(p, eps, 1.0 - eps)
        return np.log(pc / (1.0 - pc))

    def train_shadow_models(
        self,
        shadow_member_embeddings: np.ndarray,
        shadow_non_member_embeddings: np.ndarray,
    ) -> None:
        """Train bootstrap shadow classifiers and estimate global null moments."""
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

        rng = np.random.default_rng(seed=None)

        for model in self.shadow_models:
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

        all_phi: list[np.ndarray] = []
        for model in self.shadow_models:
            probs = model.predict_proba(x_n)[:, 1]
            all_phi.append(self._logit(probs))
        pooled = np.concatenate(all_phi)
        self.mu_out = float(np.mean(pooled))
        self.sigma_out = float(np.std(pooled, ddof=0))
        if self.sigma_out < 1e-8:
            # Near-degenerate null variance — keep Z-scores finite.
            self.sigma_out = 1e-8

    def evaluate_tpr_at_fpr(
        self,
        target_embeddings: np.ndarray,
        target_labels: np.ndarray,
        fpr_threshold: float = 0.001,
    ) -> float:
        """Return TPR at the empirical FPR threshold on target non-members."""
        if not 0.0 < fpr_threshold < 1.0:
            msg = "fpr_threshold must be in (0, 1)."
            raise ValueError(msg)
        x = np.asarray(target_embeddings, dtype=np.float64)
        labels = np.asarray(target_labels, dtype=np.int64).ravel()
        if x.ndim != 2 or x.shape[0] != labels.shape[0]:
            msg = "target_embeddings and target_labels must align in row count."
            raise ValueError(msg)
        avg_probs = np.mean(
            np.stack([m.predict_proba(x)[:, 1] for m in self.shadow_models], axis=0),
            axis=0,
        )
        phi_p = self._logit(avg_probs)
        z_scores = (phi_p - self.mu_out) / self.sigma_out
        p_values = 1.0 - scipy.stats.norm.cdf(z_scores)
        nm = p_values[labels == 0]
        mem = p_values[labels == 1]
        if nm.size == 0 or mem.size == 0:
            return 0.0
        tau = float(np.quantile(nm, fpr_threshold))
        return float(np.mean(mem < tau))
