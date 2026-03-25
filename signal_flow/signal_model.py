"""
Stochastic interpolants with **data-dependent coupling** (VISp → VISrl), following
Albergo et al., *Stochastic Interpolants with Data-Dependent Couplings*, ICML 2024
(arXiv:2310.03725).

**Model & training — Algorithm 1** (paper; linear interpolant, γ=0):
    I_t = α_t x_0 + β_t x_1   with   α_t = 1 − t,  β_t = t
    İ_t = \\dot{α}_t x_0 + \\dot{β}_t x_1 = x_1 − x_0
    Coupling: paired (x_0, x_1) from training data (VISp / VISrl z-scores, same trial).
    Objective (minibatch Monte Carlo of Eq. (7) in the paper):
        L̂_b = n_b^{-1} Σ_i [ ‖\\hat{b}_{t_i}(I_{t_i})‖² − 2 \\dot{I}_{t_i} · \\hat{b}_{t_i}(I_{t_i}) ]

**Sampling — Algorithm 2** (forward Euler):
    X_0 = source embedding (VISp test z); optional Gaussian jitter matches x_0 = m(x_1)+σζ when σ>0.
    X_{n+1} = X_n + N^{-1} \\hat{b}_{n/N}(X_n),  n = 0, …, N−1

Reference: https://arxiv.org/pdf/2310.03725
"""

from __future__ import annotations

import torch
import torch.nn as nn


def sincos_embed(t: torch.Tensor, time_dimension: int, base: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding: (B,) -> (B, time_dimension)."""
    half = time_dimension // 2
    k = torch.arange(half, device=t.device, dtype=t.dtype)
    w_k = base ** (-2 * k / time_dimension)
    angles = t[:, None] * w_k[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class MLP_Residual(nn.Module):
    """
    Residual MLP \\hat{b}(x_t, t); same structure as gaussian_model.MLP_Residual.
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


def interpolant_state(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    I_t = (1 - t) x0 + t x1  (Algorithm 1; α_t=1-t, β_t=t, γ=0).
    x0, x1: [B, D]; t: [B]
    """
    tb = t[:, None]
    return (1.0 - tb) * x0 + tb * x1


def interpolant_velocity(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """İ_t = x1 - x0 for the linear interpolant above."""
    return x1 - x0


def training_loss_algorithm1(model: MLP_Residual, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """
    Algorithm 1 (VISp→VISrl bridge): paired x0, x1; t ~ U(0,1); loss L̂_b on velocity network.

    Args:
        x0: [B, D] e.g. standardized VISp
        x1: [B, D] e.g. standardized VISrl (same indices as x0)
    """
    t = torch.rand(x0.shape[0], device=x0.device, dtype=x0.dtype)
    x_t = interpolant_state(x0, x1, t)
    i_dot = interpolant_velocity(x0, x1)
    b = model(x_t, t)
    # L̂_b = mean_i [ ||b||^2 - 2 i_dot · b ]
    return (b.pow(2).sum(dim=-1) - 2.0 * (i_dot * b).sum(dim=-1)).mean()


def cfm_loss_bridge(model: MLP_Residual, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Alias for `training_loss_algorithm1` (paired VISp / VISrl bridge)."""
    return training_loss_algorithm1(model, x0, x1)


def cfm_loss(model: MLP_Residual, x1: torch.Tensor) -> torch.Tensor:
    """
    Independent coupling: x0 ~ N(0, I), x1 targets (noise → data), same linear interpolant + Algorithm 1 loss.
    """
    x0 = torch.randn_like(x1)
    return training_loss_algorithm1(model, x0, x1)


@torch.no_grad()
def integrate_euler_algorithm2(
    model: MLP_Residual,
    x_init: torch.Tensor,
    n_steps: int = 100,
    *,
    noise_std: float = 0.0,
) -> torch.Tensor:
    """
    Algorithm 2: forward Euler. X_{n+1} = X_n + N^{-1} \\hat{b}_{n/N}(X_n), t = n/N.

    If noise_std > 0, use X_0 = x_init + noise_std * ζ (optional stochasticity at the source).
    """
    model.eval()
    x = x_init.clone()
    if noise_std > 0.0:
        x = x + noise_std * torch.randn_like(x)
    device, dtype = x.device, x.dtype
    n = x.shape[0]
    inv_N = 1.0 / n_steps
    for step in range(n_steps):
        t = torch.full((n,), step / n_steps, device=device, dtype=dtype)
        x = x + model(x, t) * inv_N
    return x


@torch.no_grad()
def integrate_euler(
    model: MLP_Residual,
    x_init: torch.Tensor,
    n_steps: int = 100,
) -> torch.Tensor:
    """Backward-compatible name; delegates to Algorithm 2 Euler."""
    return integrate_euler_algorithm2(model, x_init, n_steps=n_steps, noise_std=0.0)


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
    """Unconditional sample: x0 ~ N(0,I), then Algorithm 2."""
    x0 = torch.randn(n_samples, data_dim, device=device, dtype=dtype)
    return integrate_euler_algorithm2(model, x0, n_steps=n_steps)


def frechet_embedding_distance(
    real: torch.Tensor | "np.ndarray",
    fake: torch.Tensor | "np.ndarray",
    eps: float = 1e-6,
) -> float:
    """
    Fréchet distance between two batches of vectors (embedding \"FID\" / FD).

    Treats features as multivariate Gaussians (mean + covariance); returns d^2.
    Use sqrt for same scale as classical FID if desired.

    Args:
        real, fake: [N, D] (numpy or torch)
    """
    import numpy as np

    def _to_numpy(a):
        if torch.is_tensor(a):
            return a.detach().cpu().float().numpy()
        return np.asarray(a, dtype=np.float64)

    x = _to_numpy(real)
    y = _to_numpy(fake)
    mu1 = x.mean(axis=0)
    mu2 = y.mean(axis=0)
    c1 = np.cov(x, rowvar=False)
    c2 = np.cov(y, rowvar=False)
    if c1.ndim == 0:
        c1 = np.array([[float(c1)]])
    if c2.ndim == 0:
        c2 = np.array([[float(c2)]])
    c1 = c1 + np.eye(c1.shape[0]) * eps
    c2 = c2 + np.eye(c2.shape[0]) * eps

    diff = mu1 - mu2
    mean_term = float(diff @ diff)

    try:
        from scipy.linalg import sqrtm as matrix_sqrtm
    except ImportError:  # pragma: no cover
        matrix_sqrtm = None
    if matrix_sqrtm is not None:
        covmean = matrix_sqrtm(c1 @ c2)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        trace_term = np.trace(c1 + c2 - 2.0 * covmean)
    else:
        # Fallback: eigendecomposition of c1 @ c2
        prod = (c1 @ c2).astype(np.float64)
        w, v = np.linalg.eig(prod)
        w = np.maximum(np.real(w), 0.0)
        sqrt_prod = (v @ np.diag(np.sqrt(w + 0j)) @ np.linalg.inv(v + 0j)).real
        trace_term = float(np.trace(c1 + c2 - 2.0 * sqrt_prod))
    return float(mean_term + trace_term)
