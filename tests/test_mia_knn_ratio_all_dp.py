"""Integration check that k-NN ratio MIA accepts every DP mechanism output."""
from __future__ import annotations
import numpy as np
import torch
from src.evaluation.mia_knn_ratio import KNNDistanceRatioMembershipInference
from src.models.central_dp import CentralDPMechanism
from src.models.local_dp import LocalDPProjector
from src.models.metric_dp import MetricDPMechanism
def _unit_rows(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)
def test_knn_ratio_runs_on_baseline_central_local_and_metric_dp() -> None:
    """The added MIA baseline should produce finite TPRs for all mechanisms."""
    rng = np.random.default_rng(123)
    shadow_members = _unit_rows(rng.normal(size=(30, 384)))
    shadow_non_members = _unit_rows(rng.normal(size=(30, 384)))
    target = np.vstack([shadow_members[:10], shadow_non_members[:10]])
    labels = np.concatenate(
        [np.ones(10, dtype=np.int64), np.zeros(10, dtype=np.int64)]
    )
    clean_corpus = np.vstack([shadow_members, shadow_non_members])
    attack = KNNDistanceRatioMembershipInference(k=5).fit(
        shadow_members, shadow_non_members
    )
    central = CentralDPMechanism(epsilon=1.0, delta=1e-5).apply_noise(target)
    local = LocalDPProjector(
        input_dim=384, bottleneck_dim=16, epsilon=1.0, delta=1e-5
    )
    local.eval()
    with torch.no_grad():
        local_dp = local(torch.from_numpy(target.astype(np.float32))).numpy()
    metric = MetricDPMechanism(epsilon=1.0, delta=1e-5, k=5).apply_noise(
        target, clean_corpus
    )
    outputs = {
        "Baseline": target,
        "Central": central,
        "Local": local_dp,
        "Metric": metric,
    }
    for mechanism, embeddings in outputs.items():
        tpr = attack.evaluate_tpr_at_fpr(embeddings, labels, fpr_threshold=0.1)
        assert np.isfinite(tpr), mechanism
        assert 0.0 <= tpr <= 1.0, mechanism
        assert embeddings.shape[1] == 384, mechanism
