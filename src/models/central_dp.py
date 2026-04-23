"""Central (dataset-level) differential privacy for embedding vectors (vectorized)."""

import numpy as np

# Sensitivity in L2: we assume embeddings (or the function output we privatize) live
# on a scale where a single user change is bounded. For per-row unit-sphere vectors, a
# standard bound used with ``eps``-``delta`` analytical Gaussian is ``Delta_2 = 2.0``:
# two points on the unit ball can be at most Euclidean distance 2. The mechanism below
# uses that value inside ``sigma`` (see the factor ``2.0`` in the numerator), matching
# the spec for ``(epsilon, delta)``-DP with analytical Gaussian noise.


class CentralDPMechanism:
    """Apply a vectorized analytical Gaussian mechanism to a batch of embeddings.

    This bypasses a slow per-coordinate loop (e.g. in reference libraries) and matches
    the standard closed-form std-dev for the Gaussian mechanism at fixed ``(epsilon,
    delta)`` with L2-sensitivity 2.0 (see implementation comments in ``apply_noise``).

    Attributes:
        epsilon: Privacy parameter ``epsilon`` in ``(epsilon, delta)``-DP.
        delta: Privacy parameter ``delta``; must be in ``(0, 1)`` and small enough
            that ``log(1.25 / delta)`` is finite.
    """

    def __init__(self, epsilon: float, delta: float = 1e-5) -> None:
        """Store target privacy parameters for repeated ``apply_noise`` calls.

        Args:
            epsilon: Target ``epsilon`` in ``(epsilon, delta)``-differential privacy.
            delta: Target ``delta``; default ``1e-5`` is common in ML literature.
        """
        if epsilon <= 0.0 or delta <= 0.0 or delta >= 1.0:
            msg = "Need epsilon > 0 and 0 < delta < 1 for the analytical noise scale."
            raise ValueError(msg)
        self.epsilon = float(epsilon)
        self.delta = float(delta)

    def apply_noise(self, embeddings: np.ndarray) -> np.ndarray:
        """Add i.i.d. Gaussian noise calibrated to ``(epsilon, delta)``-DP.

        The noise scale (per coordinate) is:

        ``sigma = (2.0 * sqrt(2.0 * log(1.25 / delta)) / epsilon``,

        which matches the user's analytical Gaussian design with *L2 sensitivity* fixed
        at 2.0. Geometrically, many DP analyses bound vector perturbations in L2; with
        unit-norm *differences* between worst-case database neighbors, the L2
        **sensitivity of a sum-style embedding** is at most 2, hence the leading
        ``2.0`` factor in ``sigma``.

        Args:
            embeddings: 2D array of shape ``(n, d)`` or 1D ``(d,)``.

        Returns:
            ``embeddings + noise`` with the same shape and dtype (float) as the input
            (after promotion to a floating dtype if integer arrays are passed; callers
            should pass float for clarity).
        """
        x = np.asarray(embeddings, dtype=np.float64)
        if x.ndim not in (1, 2):
            msg = "embeddings must be 1- or 2-dimensional."
            raise ValueError(msg)
        # Analytical std-dev: sensitivity 2.0 is baked in as required by the project.
        sigma = 2.0 * np.sqrt(2.0 * np.log(1.25 / self.delta)) / self.epsilon
        noise = np.random.normal(loc=0.0, scale=sigma, size=x.shape)
        return x + noise
