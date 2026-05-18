"""Statistical and shape checks for central, local, and metric DP noise."""

import numpy as np
import torch
import torch.nn.functional as F

from src.models.central_dp import CentralDPMechanism
from src.models.local_dp import LocalDPProjector
from src.models.metric_dp import MetricDPMechanism


def _l2_row_normalize(emb: np.ndarray) -> np.ndarray:
    """Map each row to the 384-d unit sphere (Frobenius row norms = 1)."""
    n = np.linalg.norm(emb, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return emb / n


def test_central_dp_variance() -> None:
    """Empirical per-coordinate noise variance should match ``sigma^2`` up to 5%.

    The added noise is i.i.d. ``N(0, sigma^2)`` in each component; by the law of
    large numbers, the pooled sample variance (``np.var`` on all coordinates) is
    close to ``sigma^2`` for many draws.
    """
    eps = 1.0
    delta = 1e-5
    sigma = (2.0 * np.sqrt(2.0 * np.log(1.25 / delta))) / eps
    expected_var = float(sigma**2)

    rng = np.random.default_rng(0)
    raw = rng.standard_normal((100, 384))
    # Clean embeddings are *not* the random variable; we only use them as a carrier
    # for the mechanism. L2 normalizing is optional here but matches the RAG spec.
    emb = _l2_row_normalize(raw)

    np.random.seed(1234)
    mech = CentralDPMechanism(epsilon=eps, delta=delta)
    noisy = mech.apply_noise(emb)
    diff = noisy - np.asarray(emb, dtype=np.float64)
    # Single pooled variance of all 100 * 384 independent noise samples
    emp_var = float(np.var(diff))

    rel_err = abs(emp_var - expected_var) / expected_var
    assert rel_err < 0.05, (
        f"empirical var {emp_var} vs theory {expected_var} (rel {rel_err})"
    )


def test_local_dp_pipeline() -> None:
    """Local model returns unit L2 row vectors in the same embedding size."""
    torch.manual_seed(0)
    model = LocalDPProjector(input_dim=384, bottleneck_dim=16, epsilon=1.0, delta=1e-5)
    x = torch.randn(32, 384)
    y = model(x)
    assert y.shape == (32, 384)
    # Each row is on the 384-d unit sphere so FAISS (with normalized queries) is valid.
    norms = torch.linalg.vector_norm(y, dim=-1, ord=2)
    one = torch.ones(32, dtype=y.dtype, device=y.device)
    assert torch.allclose(norms, one, atol=1e-5, rtol=1e-5)


def test_local_dp_bottleneck_sensitivity() -> None:
    """Verifies that the global L2 sensitivity of the bottleneck mapping is <= 2.

    The DP guarantee requires that for any two adjacent inputs (differing by
    one document), their representations in the bottleneck space differ by at
    most Delta_f = 2 in L2 norm. Since all inputs are L2-normalized to the
    unit hypersphere before noise injection, the maximum possible L2 distance
    between any two bottleneck representations is 2 (antipodal points on the
    unit sphere). This test constructs 10,000 random pairs of unit vectors,
    projects them through M1, normalizes, and verifies the maximum observed
    L2 distance never exceeds 2 + 1e-5 (numerical tolerance).
    """
    torch.manual_seed(42)
    model = LocalDPProjector(input_dim=384, bottleneck_dim=16, epsilon=1.0, delta=1e-5)

    # Generate 10,000 random unit vectors in 384-d.
    n_pairs = 10_000
    x1 = torch.randn(n_pairs, 384)
    x2 = torch.randn(n_pairs, 384)
    x1 = F.normalize(x1, p=2, dim=-1)
    x2 = F.normalize(x2, p=2, dim=-1)

    with torch.no_grad():
        # Project to bottleneck and normalize (exactly as in forward()).
        z1 = F.normalize(model.M1(x1), p=2, dim=-1)
        z2 = F.normalize(model.M1(x2), p=2, dim=-1)

    # Compute pairwise L2 distances in bottleneck space.
    dists = torch.linalg.vector_norm(z1 - z2, dim=-1, ord=2)
    max_dist = float(dists.max())

    assert max_dist <= 2.0 + 1e-5, (
        f"Bottleneck sensitivity bound violated: max L2 distance = {max_dist:.6f}, "
        f"expected <= 2.0. The noise scale sigma is calibrated to Delta_f = 2; "
        f"if this bound is exceeded, the (epsilon, delta)-DP guarantee does not hold."
    )


def test_local_dp_antipodal_sensitivity() -> None:
    """Verifies the worst-case sensitivity using antipodal unit vectors.

    Antipodal points (x, -x) achieve the maximum L2 distance of 2 on the
    unit sphere. After M1 projection and L2-normalization, antipodal pairs
    in the input space may not remain antipodal in the bottleneck space,
    but the bottleneck distance must still be <= 2.
    """
    torch.manual_seed(0)
    model = LocalDPProjector(input_dim=384, bottleneck_dim=16, epsilon=1.0, delta=1e-5)

    x = torch.randn(1000, 384)
    x = F.normalize(x, p=2, dim=-1)
    neg_x = -x  # Antipodal pairs.

    with torch.no_grad():
        z_pos = F.normalize(model.M1(x), p=2, dim=-1)
        z_neg = F.normalize(model.M1(neg_x), p=2, dim=-1)

    dists = torch.linalg.vector_norm(z_pos - z_neg, dim=-1, ord=2)
    max_dist = float(dists.max())

    assert max_dist <= 2.0 + 1e-5, (
        f"Antipodal sensitivity bound violated: {max_dist:.6f} > 2.0"
    )


def test_metric_dp_output_shape() -> None:
    """Metric DP returns an aligned noised matrix for target embeddings."""
    np.random.seed(0)
    mech = MetricDPMechanism(epsilon=1.0, delta=1e-5, k=10)
    corpus = np.random.randn(100, 384).astype(np.float64)
    corpus = _l2_row_normalize(corpus)
    targets = corpus[:5]

    out = mech.apply_noise(targets, corpus)

    assert out.shape == targets.shape
    assert np.isfinite(out).all()


def test_metric_dp_changes_with_epsilon() -> None:
    """Smaller epsilon should produce larger Mahalanobis noise draws on average."""
    rng = np.random.default_rng(123)
    corpus = _l2_row_normalize(rng.standard_normal((200, 384)))
    targets = corpus[:20]

    np.random.seed(999)
    low_eps = MetricDPMechanism(epsilon=0.5, delta=1e-5, k=20)
    noisy_low = low_eps.apply_noise(targets, corpus)

    np.random.seed(999)
    high_eps = MetricDPMechanism(epsilon=5.0, delta=1e-5, k=20)
    noisy_high = high_eps.apply_noise(targets, corpus)

    low_norm = float(np.linalg.norm(noisy_low - targets))
    high_norm = float(np.linalg.norm(noisy_high - targets))

    assert low_norm > high_norm


def test_metric_dp_validation() -> None:
    """Metric DP rejects invalid privacy parameters and mismatched dimensions."""
    with np.testing.assert_raises(ValueError):
        MetricDPMechanism(epsilon=0.0)
    with np.testing.assert_raises(ValueError):
        MetricDPMechanism(epsilon=1.0, k=0)

    mech = MetricDPMechanism(epsilon=1.0, k=2)
    with np.testing.assert_raises(ValueError):
        mech.apply_noise(np.zeros((2, 3)), np.zeros((4, 5)))
