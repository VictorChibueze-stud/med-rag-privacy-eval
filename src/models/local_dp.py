import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalDPProjector(nn.Module):
    def __init__(self, input_dim: int=384, bottleneck_dim: int=16, epsilon: float=1.0, delta: float=1e-05) -> None:
        if epsilon <= 0.0 or delta <= 0.0 or delta >= 1.0:
            msg = 'Need epsilon > 0 and 0 < delta < 1.'
            raise ValueError(msg)
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.M1 = nn.Linear(input_dim, bottleneck_dim, bias=False)
        self.M2 = nn.Linear(bottleneck_dim, input_dim, bias=False)
        nn.init.orthogonal_(self.M1.weight)
        nn.init.orthogonal_(self.M2.weight)
        self.M1.weight.requires_grad = False
        self.M2.weight.requires_grad = False

    def inject_noise(self, z: torch.Tensor) -> torch.Tensor:
        sigma = 2.0 * np.sqrt(2.0 * np.log(1.25 / self.delta)) / self.epsilon
        return z + torch.randn_like(z) * float(sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.normalize(self.M1(x), p=2, dim=-1)
        y = self.M2(self.inject_noise(z))
        return F.normalize(y, p=2, dim=-1)
