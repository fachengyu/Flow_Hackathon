"""
Residual MLP vector field for conditional flow matching on the synthetic
Gaussian dataset using the Optimal Transport (OT) conditional path.

Conditional probability path:
    psi_t(x0) = (1 - (1 - sigma_min) * t) * x0 + t * x1

CFM loss (OT, eq. 23):
    L = E_{t, x0, x1} || u_theta(psi_t(x0), t) - (x1 - (1 - sigma_min) * x0) ||^2
"""

import torch
import torch.nn as nn

from gaussian_data import DIM

SIGMA_MIN = 1e-4


def sincos_embed(t: torch.Tensor, time_dimension: int, base: float = 10000.0) -> torch.Tensor:
    """Sinusoidal positional embedding for scalar time values, shape (B,) -> (B, time_dimension)."""
    half   = time_dimension // 2
    k      = torch.arange(half, device=t.device, dtype=t.dtype)
    w_k    = base ** (-2 * k / time_dimension)
    angles = t[:, None] * w_k[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class MLP_Residual(nn.Module):
    """
    Residual MLP vector field u_theta(x_t, t).

    Maps R^input_size x R^time_dimension -> R^output_size.
    Time is embedded via sinusoidal encoding and injected
    at each hidden layer via a learned projection.
    """

    def __init__(self, input_size, hidden_size, amount_layers, output_size, time_dimension):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dimension, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.H   = hidden_size
        self.L   = amount_layers
        self.d_t = time_dimension

        self.input_layer  = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(amount_layers)
        ])
        self.time_projs = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(amount_layers)
        ])
        self.out_layer = nn.Linear(hidden_size, output_size)
        self.act       = nn.SiLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        phi_t = sincos_embed(t, self.d_t)
        e_t   = self.time_mlp(phi_t)

        h = self.input_layer(x)

        for layer, proj in zip(self.hidden_layers, self.time_projs):
            h = h + self.act(layer(h) + proj(e_t))

        return self.out_layer(h)


def cfm_loss(model: MLP_Residual, x1: torch.Tensor) -> torch.Tensor:
    """
    OT conditional flow matching loss.

    x1: [B, DIM] samples from the target N(0, Sigma)
    """
    x0 = torch.randn_like(x1)
    t  = torch.rand(x1.shape[0], device=x1.device)

    # OT interpolation: psi_t(x0) = (1 - (1 - sigma_min) * t) * x0 + t * x1
    sigma_t = 1 - (1 - SIGMA_MIN) * t[:, None]
    x_t     = sigma_t * x0 + t[:, None] * x1

    # OT target velocity: x1 - (1 - sigma_min) * x0
    target  = x1 - (1 - SIGMA_MIN) * x0

    return ((model(x_t, t) - target) ** 2).mean()


@torch.no_grad()
def sample(model: MLP_Residual, n_samples: int,
           n_steps: int = 100, device: str = "cpu") -> torch.Tensor:
    """
    Euler integration from x0 ~ N(0, I) to x1 ~ N(0, Sigma).

    Returns: [n_samples, DIM]
    """
    model.eval()
    x  = torch.randn(n_samples, DIM, device=device)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((n_samples,), i / n_steps, device=device)
        x = x + model(x, t) * dt
    return x