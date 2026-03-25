"""
Residual MLP vector field for conditional flow matching on the synthetic
Gaussian dataset using the Optimal Transport (OT) conditional path.

Conditional probability path:
    psi_t(x0) = (1 - (1 - sigma_min) * t) * x0 + t * x1

CFM loss:
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

    # EXERCISE 1

    """
    TODO: Implement the OT conditional flow matching loss.

    x1: [B, DIM] samples from the target distribution

    Steps:
    1. Sample x0 ~ N(0, I) with same shape as x1
    2. Sample t ~ Uniform(0,1) for each batch element
    3. Construct the interpolation x_t = psi_t(x0)
    4. Compute the target velocity u_t
    5. Return mean squared error between model(x_t, t) and u_t
    """

    # TODO: sample base distribution
    x0 = torch.randn_like(x1)

    # TODO: sample time
    t = torch.rand(x1.shape[0], device=x1.device)

    # TODO: compute sigma_t = 1 - (1 - SIGMA_MIN) * t
    sigma_t = 1 - (1 - SIGMA_MIN) * t[:, None]

    # TODO: compute interpolated point x_t
    x_t = sigma_t*x0 + t[:,None]*x1

    # TODO: compute target velocity
    target = x1 - (1-SIGMA_MIN)*x0

    # TODO: compute and return loss
    loss = ((model(x_t, t) - target)**2).mean()
    return loss


@torch.no_grad()
def sample(model: MLP_Residual, n_samples: int,
           n_steps: int = 100, device: str = "cpu") -> torch.Tensor:
    """
    Exercise 3: Sample from the learned flow using Euler integration.

    Goal:
    Start from x0 ~ N(0, I), evolve the samples forward in time using the
    learned vector field, and return the final samples.

    Steps:
    1. Put the model in evaluation mode
    2. Sample the initial points x from N(0, I)
    3. Set the Euler step size dt = 1 / n_steps
    4. For each step i, create the time vector t
    5. Update x using the Euler rule x <- x + model(x, t) * dt
    6. Return the final samples
    """

    # TODO (Exercise 3): put the model in evaluation mode
    model.eval()

    # TODO (Exercise 3): sample initial points x of shape [n_samples, DIM]
    x = torch.randn(n_samples, DIM, device=device)

    # TODO (Exercise 3): set the Euler step size
    dt = 1.0 / n_steps

    for i in range(n_steps):
        # TODO (Exercise 3): create the time vector of shape [n_samples]
        t = torch.full((n_samples,), i / n_steps, device=device)

        # TODO (Exercise 3): perform one Euler update step
        x = x + model(x, t) * dt

    return x