"""
Flow-Based Distribution Alignment (FDA) — flow matching transport for neural embeddings.

Reference:
  Puli Wang, Yu Qi, Yueming Wang, Gang Pan (2025).
  "Flow Matching for Few-Trial Neural Adaptation with Stable Latent Dynamics."
  ICML 2025. OpenReview: https://openreview.net/pdf?id=nKJEAQ6JCY
  Code: https://github.com/wangpuli/FDA

This module ports the **linear (IC) and GVP probability paths** from FDA’s SiT transport
(`flow/transport/path.py` in the official repo) and trains a **velocity field** MLP with the
same conditional flow-matching objective as `Transport.training_losses` with
`ModelType.VELOCITY` (see `flow/transport/transport.py`).

**Setup (FDA / SiT convention, paper-aligned):**
  - `x0`: **simple latent** — standard Gaussian noise `N(0, I)` (same dimension as neural
    embeddings), matching FDA’s generative pre-training style (Wang et al., ICML 2025).
  - `x1`: **target neural representation** to decode (e.g. standardized VISrl or VISp Ca
    embedding for that trial).
  - `c` (**condition**): **observed neural window** for the same trial — e.g. standardized
    VISp activity when adapting to VISrl (`x1`), or `x1` plus Gaussian noise when modelling a
    single area (so the flow is non-trivial at training time).
  - Path: `x_t = α(t) x1 + σ(t) x0`,  target velocity `u_t = α'(t) x1 + σ'(t) x0` (unchanged).
  - Model: **conditional** velocity `v_θ(x_t, t, c)`; flow matching still regresses to `u_t`.

  - **ICPlan (linear):** `α(t)=t`, `σ(t)=1-t`  ⇒  `x_t = t x1 + (1-t) x0`,  `u_t = x1 - x0`.
  - **GVPCPlan:** `α(t)=sin(πt/2)`, `σ(t)=cos(πt/2)` (as in FDA `flow/transport/path.py`).

**Inference:** sample `z ~ N(0,I)`, set `c` to the observed test window, integrate
`dX/dt = v_θ(X, t, c)` from `t=0` to `t=1` to obtain a representation for decoding.

The unconditional `VelocityFieldMLP` + `fda_flow_matching_loss` remain for ablations
(empirical endpoint transport without latent noise or context).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Path plans (from FDA / SiT `flow.transport.path`)
# ---------------------------------------------------------------------------


def expand_t_like_x(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape time `t` [B] to broadcast with `x` [B, *]."""
    if t.dim() > 1:
        t = t.view(t.shape[0])
    dims = [1] * (x.dim() - 1)
    return t.view(t.shape[0], *dims)


class ICPlan:
    """
    Linear coupling plan (IC). Same as FDA `path.ICPlan` with tensor-valued derivatives.
    x_t = t * x1 + (1 - t) * x0,  u_t = x1 - x0.
    """

    def compute_alpha_t(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return t, torch.ones_like(t)

    def compute_sigma_t(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return 1.0 - t, -torch.ones_like(t)

    def compute_mu_t(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        t_b = expand_t_like_x(t, x1)
        alpha_t, _ = self.compute_alpha_t(t_b)
        sigma_t, _ = self.compute_sigma_t(t_b)
        return alpha_t * x1 + sigma_t * x0

    def compute_ut(
        self,
        t: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
        xt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t_b = expand_t_like_x(t, x1)
        _, d_alpha_t = self.compute_alpha_t(t_b)
        _, d_sigma_t = self.compute_sigma_t(t_b)
        return d_alpha_t * x1 + d_sigma_t * x0


class GVPCPlan(ICPlan):
    """
    GVP path from FDA `path.GVPCPlan`:
    α(t) = sin(π t / 2),  σ(t) = cos(π t / 2).
    """

    def compute_alpha_t(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha_t = torch.sin(t * (math.pi / 2))
        d_alpha_t = (math.pi / 2) * torch.cos(t * (math.pi / 2))
        return alpha_t, d_alpha_t

    def compute_sigma_t(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sigma_t = torch.cos(t * (math.pi / 2))
        d_sigma_t = -(math.pi / 2) * torch.sin(t * (math.pi / 2))
        return sigma_t, d_sigma_t


# ---------------------------------------------------------------------------
# Time embedding + velocity MLP (replaces FDA SiT backbone for vector data)
# ---------------------------------------------------------------------------


class SinusoidalTimeEmbed(nn.Module):
    """Sinusoidal features + small MLP (scalar t ∈ [0,1] → cond_dim)."""

    def __init__(self, cond_dim: int, freq_dim: int = 128):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    @staticmethod
    def _sinusoidal(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / max(half - 1, 1)
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() > 1:
            t = t.flatten()
        emb = self._sinusoidal(t, self.freq_dim)
        return self.mlp(emb)


class VelocityFieldMLP(nn.Module):
    """Unconditional MLP velocity v_θ(x, t) for dX/dt = v_θ (ablation / legacy transport)."""

    def __init__(
        self,
        data_dim: int,
        hidden_dim: int = 512,
        cond_dim: int = 128,
        n_layers: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.time_emb = SinusoidalTimeEmbed(cond_dim)

        layers: list[nn.Module] = []
        in_dim = data_dim + cond_dim
        for i in range(n_layers):
            out_d = hidden_dim
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, out_d))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, data_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() > 1:
            t = t.squeeze(-1)
        te = self.time_emb(t)
        h = self.backbone(torch.cat([x, te], dim=-1))
        return self.out(h)


class ConditionalVelocityFieldMLP(nn.Module):
    """
    FDA-style **conditional** velocity v_θ(x, t, c) for dX/dt = v_θ(X, t, c).

    `c` encodes the observed neural window (same dimensionality as `x` here: one Ca
    embedding vector per trial). Matches SiT/FDA pattern of conditioning the drift on
    side information while `x` flows from noise to data along the probability path.
    """

    def __init__(
        self,
        data_dim: int,
        hidden_dim: int = 512,
        time_emb_dim: int = 128,
        ctx_dim: int = 128,
        n_layers: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.time_emb = SinusoidalTimeEmbed(time_emb_dim)
        self.ctx_encoder = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, ctx_dim),
        )
        layers: list[nn.Module] = []
        in_dim = data_dim + time_emb_dim + ctx_dim
        for i in range(n_layers):
            out_d = hidden_dim
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, out_d))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, data_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if t.dim() > 1:
            t = t.squeeze(-1)
        te = self.time_emb(t)
        ce = self.ctx_encoder(cond)
        h = self.backbone(torch.cat([x, te, ce], dim=-1))
        return self.out(h)


# ---------------------------------------------------------------------------
# FDA-style flow-matching loss (matches Transport + VELOCITY in FDA repo)
# ---------------------------------------------------------------------------


def fda_flow_matching_loss(
    model: VelocityFieldMLP,
    x0: torch.Tensor,
    x1: torch.Tensor,
    plan: Union[ICPlan, GVPCPlan],
    train_eps: float = 1e-5,
) -> torch.Tensor:
    """
    Sample t ~ Uniform[eps, 1-eps], build (x_t, u_t) via the path plan, regress MSE(v_θ, u_t).

    `train_eps` mirrors FDA `Transport.check_interval` stability near t ∈ {0, 1}.
    """
    b = x0.shape[0]
    device = x0.device
    t0 = train_eps
    t1 = 1.0 - train_eps
    t = torch.rand(b, device=device) * (t1 - t0) + t0

    xt = plan.compute_mu_t(t, x0, x1)
    ut = plan.compute_ut(t, x0, x1, xt)
    v_pred = model(xt, t)
    return F.mse_loss(v_pred, ut)


def fda_conditional_flow_matching_loss(
    model: ConditionalVelocityFieldMLP,
    x0: torch.Tensor,
    x1: torch.Tensor,
    cond: torch.Tensor,
    plan: Union[ICPlan, GVPCPlan],
    train_eps: float = 1e-5,
) -> torch.Tensor:
    """
    Conditional flow matching: same bridge (x_t, u_t) from (x0, x1), but
    v_θ = model(x_t, t, cond). `cond` is the observed neural context (not interpolated).
    """
    b = x0.shape[0]
    device = x0.device
    t0 = train_eps
    t1 = 1.0 - train_eps
    t = torch.rand(b, device=device) * (t1 - t0) + t0

    xt = plan.compute_mu_t(t, x0, x1)
    ut = plan.compute_ut(t, x0, x1, xt)
    v_pred = model(xt, t, cond)
    return F.mse_loss(v_pred, ut)


@torch.no_grad()
def integrate_euler(
    model: VelocityFieldMLP,
    x_init: torch.Tensor,
    n_steps: int = 100,
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> torch.Tensor:
    """Euler ODE: dX/dt = v_θ(X, t)."""
    model.eval()
    device = x_init.device
    x = x_init.clone()
    dt = (t_end - t_start) / n_steps
    for k in range(n_steps):
        t = torch.full((x.shape[0],), t_start + dt * k, device=device, dtype=x.dtype)
        x = x + dt * model(x, t)
    return x


@torch.no_grad()
def integrate_rk4(
    model: VelocityFieldMLP,
    x_init: torch.Tensor,
    n_steps: int = 64,
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> torch.Tensor:
    """RK4 integration of dX/dt = v_θ(X, t)."""
    model.eval()
    device = x_init.device
    x = x_init.clone()
    dt = (t_end - t_start) / n_steps

    def v_field(z: torch.Tensor, t_scalar: float) -> torch.Tensor:
        tt = torch.full((z.shape[0],), t_scalar, device=device, dtype=z.dtype)
        return model(z, tt)

    t = t_start
    for _ in range(n_steps):
        k1 = v_field(x, t)
        k2 = v_field(x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = v_field(x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = v_field(x + dt * k3, t + dt)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
    return x


@torch.no_grad()
def integrate_rk4_conditional(
    model: ConditionalVelocityFieldMLP,
    x_init: torch.Tensor,
    cond: torch.Tensor,
    n_steps: int = 64,
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> torch.Tensor:
    """RK4 for dX/dt = v_θ(X, t, cond) with fixed `cond` along the trajectory."""
    model.eval()
    device = x_init.device
    x = x_init.clone()
    dt = (t_end - t_start) / n_steps

    def v_field(z: torch.Tensor, t_scalar: float) -> torch.Tensor:
        tt = torch.full((z.shape[0],), t_scalar, device=device, dtype=z.dtype)
        return model(z, tt, cond)

    t = t_start
    for _ in range(n_steps):
        k1 = v_field(x, t)
        k2 = v_field(x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = v_field(x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = v_field(x + dt * k3, t + dt)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
    return x


def paired_minibatches(
    x0_full: torch.Tensor,
    x1_full: torch.Tensor,
    batch_size: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random pairing of source / target minibatch indices."""
    n0 = x0_full.shape[0]
    n1 = x1_full.shape[0]
    i0 = torch.randint(0, n0, (batch_size,), generator=generator, device=x0_full.device)
    i1 = torch.randint(0, n1, (batch_size,), generator=generator, device=x1_full.device)
    return x0_full[i0], x1_full[i1]


def aligned_minibatch(
    *tensors: torch.Tensor,
    batch_size: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, ...]:
    """Same random indices for every tensor (trial-aligned VISp/VISrl pairs)."""
    if not tensors:
        raise ValueError("aligned_minibatch needs at least one tensor")
    n0 = tensors[0].shape[0]
    for t in tensors[1:]:
        if t.shape[0] != n0:
            raise ValueError("aligned_minibatch: all tensors must share dim 0")
    idx = torch.randint(0, n0, (batch_size,), generator=generator, device=tensors[0].device)
    return tuple(t[idx] for t in tensors)


def standard_gaussian_latent(
    batch_size: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """x0 ~ N(0, I) with shape [batch_size, dim] (FDA-style source for flow matching)."""
    return torch.randn(batch_size, dim, device=device, dtype=dtype, generator=generator)


# Backward-compatible name (linear IC path is default FDA linear FM)
def flow_matching_loss(
    model: VelocityFieldMLP,
    x0: torch.Tensor,
    x1: torch.Tensor,
    sigma_bridge: float = 0.0,
    plan: Optional[Union[ICPlan, GVPCPlan]] = None,
    train_eps: float = 1e-5,
) -> torch.Tensor:
    """
    Deprecated: use `fda_flow_matching_loss(..., plan=ICPlan())`.
    If `sigma_bridge > 0`, falls back to old noisy-bridge heuristic (not in FDA).
    """
    if sigma_bridge > 0:
        b, _ = x0.shape
        device = x0.device
        t = torch.rand(b, device=device)
        t_b = t.view(-1, 1)
        eps = torch.randn_like(x0)
        gamma = sigma_bridge * torch.sqrt(t_b * (1.0 - t_b).clamp(min=1e-8))
        x_t = (1.0 - t_b) * x0 + t_b * x1 + gamma * eps
        v_star = x1 - x0
        return F.mse_loss(model(x_t, t), v_star)
    p = plan if plan is not None else ICPlan()
    return fda_flow_matching_loss(model, x0, x1, p, train_eps=train_eps)
