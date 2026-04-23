"""Local (on-device) DP via a small bottleneck network with local Gaussian noise."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# L2 noise scaling uses the same analytical ``sigma`` as the central model (L2
# sensitivity 2.0 in the high-level spec). The bottleneck activations are **first**
# projected to the **unit L2 ball** in ``bottleneck_dim``-space, so the vector change
# from a single user swap is again bounded in norm by 2 (the diameter of the image
# on the unit hypersphere), which justifies the same ``(epsilon, delta)`` calibration.


class LocalDPProjector(nn.Module):
    """Two linear maps with Gaussian noise in the (normalized) latent bottleneck.

    M1: ``R^{input_dim} -> R^{bottleneck_dim}``, M2: ``R^{bottleneck_dim} ->
    R^{input_dim}`` (no bias, per spec). The bottleneck is L2-normalized *before* local
    noise, so the noisy latent has controlled sensitivity; after M2, outputs are
    re-normalized to unit length for FAISS/KNN pipelines that expect unit vectors.

    Attributes:
        input_dim: Input embedding dimension (e.g. 384 for the baseline encoder).
        bottleneck_dim: Middle dimension (e.g. 16) where noise is applied.
        epsilon: Epsilon in ``(epsilon, delta)``-style local noise calibration.
        delta: Delta used in the closed-form ``sigma`` expression.
    """

    def __init__(
        self,
        input_dim: int = 384,
        bottleneck_dim: int = 16,
        epsilon: float = 1.0,
        delta: float = 1e-5,
    ) -> None:
        """Build M1, M2 and store the local DP privacy parameters.

        Args:
            input_dim: Dimension of input embedding vectors.
            bottleneck_dim: Latent size before the decoder and before noise.
            epsilon: Epsilon in the same analytical noise formula as the central path.
            delta: Small positive ``delta`` in ``(epsilon, delta)`` (default ``1e-5``).
        """
        if epsilon <= 0.0 or delta <= 0.0 or delta >= 1.0:
            msg = "Need epsilon > 0 and 0 < delta < 1 for a finite noise scale."
            raise ValueError(msg)
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.M1 = nn.Linear(input_dim, bottleneck_dim, bias=False)
        self.M2 = nn.Linear(bottleneck_dim, input_dim, bias=False)

    def inject_noise(self, z: torch.Tensor) -> torch.Tensor:
        """Add isotropic Gaussian noise to each row of the bottleneck ``z``.

        The standard deviation matches the spec:

        ``sigma = (2.0 * sqrt(2.0 * log(1.25 / delta)) / epsilon``,

        i.e. the same **scalar** as in ``CentralDPMechanism`` for sensitivity 2.0.

        Args:
            z: Tensor, typically shape ``(batch, bottleneck_dim)`` after M1+normalize.

        Returns:
            ``z + noise`` with the same shape and device/dtype as ``z``.
        """
        # Same scalar ``sigma`` as ``CentralDPMechanism.apply_noise`` (L2=2.0).
        sigma = (2.0 * np.sqrt(2.0 * np.log(1.25 / self.delta))) / self.epsilon
        noise = torch.randn_like(z) * float(sigma)
        return z + noise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project, normalize, noise, decode, then normalize the output to unit L2.

        1) ``z = M1(x)``.
        2) L2 normalize ``z`` *along the feature dim* (each row is on the
           ``bottleneck_dim``-sphere). **Sensitivity** of the *identity* on the sphere
           under replacement of one user (bounded in norm) is captured by a diameter-2
           set in ambient space, hence the *global* L2 sensitivity constant 2.0 in the
           ``sigma`` formula.
        3) Add calibrated Gaussian in that space.
        4) ``y = M2(z_noisy)`` and L2 normalize ``y`` to norm 1 for FAISS ingestion.

        Args:
            x: Batch of input embeddings, shape ``(batch, input_dim)``.

        Returns:
            Reconstructed, unit L2 row-normalized tensor, shape
            ``(batch, input_dim)``.
        """
        z = self.M1(x)
        # Map each row to the unit hypersphere in bottleneck space: norm = 1.
        z = F.normalize(z, p=2, dim=-1)
        z_noisy = self.inject_noise(z)
        y = self.M2(z_noisy)
        y = F.normalize(y, p=2, dim=-1)
        return y
