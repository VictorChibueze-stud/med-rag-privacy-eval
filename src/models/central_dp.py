import numpy as np

class CentralDPMechanism:
    def __init__(self, epsilon: float, delta: float=1e-05) -> None:
        if epsilon <= 0.0 or delta <= 0.0 or delta >= 1.0:
            msg = 'Need epsilon > 0 and 0 < delta < 1.'
            raise ValueError(msg)
        self.epsilon = float(epsilon)
        self.delta = float(delta)

    def apply_noise(self, embeddings: np.ndarray) -> np.ndarray:
        x = np.asarray(embeddings, dtype=np.float64)
        if x.ndim not in (1, 2):
            msg = 'embeddings must be 1- or 2-dimensional.'
            raise ValueError(msg)
        sigma = 2.0 * np.sqrt(2.0 * np.log(1.25 / self.delta)) / self.epsilon
        noise = np.random.normal(loc=0.0, scale=sigma, size=x.shape)
        return x + noise
