"""
Residual MLP vector field for flow matching on Allen Ca embeddings, using the same
**optimal-transport conditional path** (stochastic interpolant) as
`synthetic_gaussian/solution/gaussian_model.py`.

Conditional probability path (eq. 23 style):
    psi_t(x0) = (1 - (1 - sigma_min) * t) * x0 + t * x1

CFM loss:
    L = E_{t, x0, x1} || u_theta(psi_t, t) - (x1 - (1 - sigma_min) * x0) ||^2

- **Noise → target:** `x0 ~ N(0, I)`, `x1` standardized targets (see `cfm_loss`).
- **Paired bridge:** `x0`, `x1` both data, same batch pairing (see `cfm_loss_bridge`, e.g. VISp z → VISrl z).

Inference: Euler integrate `dx/dt = u_theta(x, t)` from `t=0` to `1` starting at the chosen `x0` (noise or VISp z).
"""

from __future__ import annotations

import torch
import torch.nn as nn

SIGMA_MIN = 1e-4


def sincos_embed(t: torch.Tensor, time_dimension: int, base: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding: (B,) -> (B, time_dimension)."""
    half = time_dimension // 2
    k = torch.arange(half, device=t.device, dtype=t.dtype)
    w_k = base ** (-2 * k / time_dimension)
    angles = t[:, None] * w_k[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class MLP_Residual(nn.Module):
    """
    Residual MLP u_theta(x_t, t); same structure as gaussian_model.MLP_Residual.
    Maps R^{input_size} × R^{time_dimension} -> R^{output_size}.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        amount_layers: int,
        output_size: int,
        time_dimension: int,
    ):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dimension, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.H = hidden_size
        self.L = amount_layers
        self.d_t = time_dimension

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(amount_layers)]
        )
        self.time_projs = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(amount_layers)]
        )
        self.out_layer = nn.Linear(hidden_size, output_size)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() > 1:
            t = t.squeeze(-1)
        phi_t = sincos_embed(t, self.d_t)
        e_t = self.time_mlp(phi_t)

        h = self.input_layer(x)

        for layer, proj in zip(self.hidden_layers, self.time_projs):
            h = h + self.act(layer(h) + proj(e_t))

        return self.out_layer(h)


def cfm_loss(model: MLP_Residual, x1: torch.Tensor) -> torch.Tensor:
    """
    OT conditional flow matching loss.

    x1: [B, D] target samples (e.g. standardized VISp embeddings).
    x0: [B, D] ~ N(0, I).
    """
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)

    sigma_t = 1 - (1 - SIGMA_MIN) * t[:, None]
    x_t = sigma_t * x0 + t[:, None] * x1

    target = x1 - (1 - SIGMA_MIN) * x0

    return ((model(x_t, t) - target) ** 2).mean()


def cfm_loss_bridge(model: MLP_Residual, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """
    Same OT conditional path as `cfm_loss`, but both endpoints are data (paired rows).

    x0, x1: [B, D] — e.g. standardized VISp and VISrl train embeddings with aligned indices.
    """
    t = torch.rand(x0.shape[0], device=x0.device, dtype=x0.dtype)

    sigma_t = 1 - (1 - SIGMA_MIN) * t[:, None]
    x_t = sigma_t * x0 + t[:, None] * x1

    target = x1 - (1 - SIGMA_MIN) * x0

    return ((model(x_t, t) - target) ** 2).mean()


@torch.no_grad()
def integrate_euler(
    model: MLP_Residual,
    x_init: torch.Tensor,
    n_steps: int = 100,
) -> torch.Tensor:
    """Euler integration of dx/dt = u_theta(x, t) from t=0 to 1 (same schedule as gaussian `sample`)."""
    model.eval()
    x = x_init.clone()
    device, dtype = x.device, x.dtype
    n = x.shape[0]
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((n,), i / n_steps, device=device, dtype=dtype)
        x = x + model(x, t) * dt
    return x


@torch.no_grad()
def sample(
    model: MLP_Residual,
    n_samples: int,
    data_dim: int,
    *,
    n_steps: int = 100,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample by integrating from x ~ N(0, I), same as `gaussian_model.sample` with explicit dim."""
    x = torch.randn(n_samples, data_dim, device=device, dtype=dtype)
    return integrate_euler(model, x, n_steps=n_steps)
