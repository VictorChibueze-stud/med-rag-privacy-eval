import numpy as np
import torch
import torch.nn.functional as F
from src.models.central_dp import CentralDPMechanism
from src.models.local_dp import LocalDPProjector
from src.models.metric_dp import MetricDPMechanism

def _l2_row_normalize(emb: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(emb, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return emb / n

def test_central_dp_variance() -> None:
    eps = 1.0
    delta = 1e-05
    sigma = 2.0 * np.sqrt(2.0 * np.log(1.25 / delta)) / eps
    expected_var = float(sigma ** 2)
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((100, 384))
    emb = _l2_row_normalize(raw)
    np.random.seed(1234)
    mech = CentralDPMechanism(epsilon=eps, delta=delta)
    noisy = mech.apply_noise(emb)
    diff = noisy - np.asarray(emb, dtype=np.float64)
    emp_var = float(np.var(diff))
    rel_err = abs(emp_var - expected_var) / expected_var
    assert rel_err < 0.05, f'empirical var {emp_var} vs theory {expected_var} (rel {rel_err})'

def test_local_dp_pipeline() -> None:
    torch.manual_seed(0)
    model = LocalDPProjector(input_dim=384, bottleneck_dim=16, epsilon=1.0, delta=1e-05)
    x = torch.randn(32, 384)
    y = model(x)
    assert y.shape == (32, 384)
    norms = torch.linalg.vector_norm(y, dim=-1, ord=2)
    one = torch.ones(32, dtype=y.dtype, device=y.device)
    assert torch.allclose(norms, one, atol=1e-05, rtol=1e-05)

def test_local_dp_bottleneck_sensitivity() -> None:
    torch.manual_seed(42)
    model = LocalDPProjector(input_dim=384, bottleneck_dim=16, epsilon=1.0, delta=1e-05)
    n_pairs = 10000
    x1 = torch.randn(n_pairs, 384)
    x2 = torch.randn(n_pairs, 384)
    x1 = F.normalize(x1, p=2, dim=-1)
    x2 = F.normalize(x2, p=2, dim=-1)
    with torch.no_grad():
        z1 = F.normalize(model.M1(x1), p=2, dim=-1)
        z2 = F.normalize(model.M1(x2), p=2, dim=-1)
    dists = torch.linalg.vector_norm(z1 - z2, dim=-1, ord=2)
    max_dist = float(dists.max())
    assert max_dist <= 2.0 + 1e-05, f'Bottleneck sensitivity bound violated: max L2 distance = {max_dist:.6f}, expected <= 2.0. The noise scale sigma is calibrated to Delta_f = 2; if this bound is exceeded, the (epsilon, delta)-DP guarantee does not hold.'

def test_local_dp_antipodal_sensitivity() -> None:
    torch.manual_seed(0)
    model = LocalDPProjector(input_dim=384, bottleneck_dim=16, epsilon=1.0, delta=1e-05)
    x = torch.randn(1000, 384)
    x = F.normalize(x, p=2, dim=-1)
    neg_x = -x
    with torch.no_grad():
        z_pos = F.normalize(model.M1(x), p=2, dim=-1)
        z_neg = F.normalize(model.M1(neg_x), p=2, dim=-1)
    dists = torch.linalg.vector_norm(z_pos - z_neg, dim=-1, ord=2)
    max_dist = float(dists.max())
    assert max_dist <= 2.0 + 1e-05, f'Antipodal sensitivity bound violated: {max_dist:.6f} > 2.0'

def test_metric_dp_output_shape() -> None:
    np.random.seed(0)
    mech = MetricDPMechanism(epsilon=1.0, delta=1e-05, k=10)
    corpus = np.random.randn(100, 384).astype(np.float64)
    corpus = _l2_row_normalize(corpus)
    targets = corpus[:5]
    out = mech.apply_noise(targets, corpus)
    assert out.shape == targets.shape
    assert np.isfinite(out).all()

def test_metric_dp_changes_with_epsilon() -> None:
    rng = np.random.default_rng(123)
    corpus = _l2_row_normalize(rng.standard_normal((200, 384)))
    targets = corpus[:20]
    np.random.seed(999)
    low_eps = MetricDPMechanism(epsilon=0.5, delta=1e-05, k=20)
    noisy_low = low_eps.apply_noise(targets, corpus)
    np.random.seed(999)
    high_eps = MetricDPMechanism(epsilon=5.0, delta=1e-05, k=20)
    noisy_high = high_eps.apply_noise(targets, corpus)
    low_norm = float(np.linalg.norm(noisy_low - targets))
    high_norm = float(np.linalg.norm(noisy_high - targets))
    assert low_norm > high_norm

def test_metric_dp_validation() -> None:
    with np.testing.assert_raises(ValueError):
        MetricDPMechanism(epsilon=0.0)
    with np.testing.assert_raises(ValueError):
        MetricDPMechanism(epsilon=1.0, k=0)
    mech = MetricDPMechanism(epsilon=1.0, k=2)
    with np.testing.assert_raises(ValueError):
        mech.apply_noise(np.zeros((2, 3)), np.zeros((4, 5)))
