"""Statistical and shape checks for central vs. local DP noise (Sprint 3)."""

import numpy as np
import torch

from src.models.central_dp import CentralDPMechanism
from src.models.local_dp import LocalDPProjector


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
