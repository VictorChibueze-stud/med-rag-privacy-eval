"""Tests for the k-NN distance-ratio MIA baseline."""
import numpy as np
from src.evaluation.mia_knn_ratio import KNNDistanceRatioMembershipInference
def test_knn_ratio_scores_member_like_points_higher() -> None:
    """Targets near the member bank should get larger scores."""
    shadow_members = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]], dtype=float)
    shadow_non_members = np.array([[5.0, 5.0], [5.1, 5.0], [5.0, 5.1]], dtype=float)
    attack = KNNDistanceRatioMembershipInference(k=2).fit(
        shadow_members, shadow_non_members
    )
    scores = attack.score_samples(np.array([[0.05, 0.05], [5.05, 5.05]], dtype=float))
    assert scores[0] > scores[1]
def test_knn_ratio_tpr_at_fpr_is_bounded() -> None:
    """TPR@FPR uses target non-members for its empirical threshold."""
    shadow_members = np.array([[0.0], [0.1], [0.2], [0.3]], dtype=float)
    shadow_non_members = np.array([[4.0], [4.1], [4.2], [4.3]], dtype=float)
    target = np.array([[0.05], [0.15], [4.05], [4.15]], dtype=float)
    labels = np.array([1, 1, 0, 0], dtype=int)
    attack = KNNDistanceRatioMembershipInference(k=1).fit(
        shadow_members, shadow_non_members
    )
    tpr = attack.evaluate_tpr_at_fpr(target, labels, fpr_threshold=0.5)
    assert 0.0 <= tpr <= 1.0
    assert tpr > 0.0
def test_knn_ratio_validates_inputs() -> None:
    """The attack rejects invalid parameters and unfitted scoring."""
    with np.testing.assert_raises(ValueError):
        KNNDistanceRatioMembershipInference(k=0)
    attack = KNNDistanceRatioMembershipInference(k=1)
    with np.testing.assert_raises(RuntimeError):
        attack.score_samples(np.zeros((1, 2)))
    with np.testing.assert_raises(ValueError):
        attack.fit(np.zeros((2, 3)), np.zeros((2, 4)))
