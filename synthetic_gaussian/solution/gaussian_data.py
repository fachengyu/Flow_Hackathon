"""
Synthetic Gaussian dataset with block covariance structure.
Provides dataset generation, a PyTorch Dataset class, and
evaluation metrics for flow matching.

Distribution:
    x from N(0, Sigma),  x in R^{DIM}

    Sigma_ij = 1.0   if i == j
    Sigma_ij = RHO   if pixel i and pixel j are in the same BLOCK_SIZE x BLOCK_SIZE spatial block
    Sigma_ij = 0.0   otherwise
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.stats import wasserstein_distance

# configuration
N          = 32
BLOCK_SIZE = 8
RHO        = 0.85
DIM        = N * N


def _build_sigma(N, block_size, rho):
    """Build the block covariance matrix Sigma of shape (DIM, DIM)."""
    rows      = np.arange(N).repeat(N)
    cols      = np.tile(np.arange(N), N)
    block_ids = (rows // block_size) * (N // block_size) + (cols // block_size)
    same      = block_ids[:, None] == block_ids[None, :]
    Sigma     = np.where(same, rho, 0.0).astype(np.float64)
    np.fill_diagonal(Sigma, 1.0)
    return Sigma


def get_gaussian_dataset(Sigma: np.ndarray, n_samples: int, seed: int = 42) -> np.ndarray:
    """
    Sample n_samples flat vectors from N(0, Sigma).

    Returns:
        numpy array of shape (n_samples, DIM), dtype float32
    """
    L = np.linalg.cholesky(Sigma)
    rng = np.random.default_rng(seed)
    Z   = rng.standard_normal((DIM, n_samples))
    X   = (L @ Z).T
    return X.astype(np.float32)


class GaussianDataset(Dataset):
    """
    PyTorch Dataset of flat vectors from N(0, Sigma).
    Each item is shape (DIM,).
    """

    def __init__(self, n_samples: int = 10000, seed: int = 42):
        data      = get_gaussian_dataset(n_samples, seed)
        self.data = torch.from_numpy(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# evaluation metric

def covariance_error(Sigma: np.ndarray, generated: np.ndarray) -> dict:
    """
    Compare the empirical covariance of generated samples to the true Sigma.

    Args:
        generated: (n_samples, DIM) numpy array of generated samples

    Returns:
        dict with keys:
            mean_abs_error   - mean |Sigma_true - Sigma_learned|
            max_abs_error    - max  |Sigma_true - Sigma_learned|
            frobenius_error  - ||Sigma_true - Sigma_learned||_F
    """
    Sigma_learned = np.cov(generated.T)
    diff          = Sigma - Sigma_learned
    return {
        "mean_abs_error":  float(np.abs(diff).mean()),
        "max_abs_error":   float(np.abs(diff).max()),
        "frobenius_error": float(np.linalg.norm(diff, "fro")),
    }


def wasserstein_error(generated: np.ndarray, n_real: int = 5000, seed: int = 0) -> dict:
    """
    Estimate the 1D marginal Wasserstein distance between generated samples
    and true samples from N(0, Sigma), averaged over all DIM dimensions.

    Args:
        generated: (n_samples, DIM) numpy array of generated samples
        n_real:    number of true samples to draw for comparison
        seed:      random seed for true samples

    Returns:
        dict with keys:
            mean_wasserstein  - average 1D Wasserstein across all dimensions
            max_wasserstein   - worst dimension
    """
    real      = get_gaussian_dataset(n_real, seed=seed)
    distances = np.array([
        wasserstein_distance(real[:, d], generated[:, d])
        for d in range(DIM)
    ])
    return {
        "mean_wasserstein": float(distances.mean()),
        "max_wasserstein":  float(distances.max()),
    }


def evaluate(generated, Sigma, n_real: int = 5000) -> dict:
    """
    Run all evaluation metrics on a set of generated samples.

    Args:
        generated: (n_samples, DIM) numpy array or torch.Tensor
        n_real:    number of real samples to use for Wasserstein comparison

    Returns:
        dict combining covariance_error and wasserstein_error results

    Example:
        generated = flow_model.sample(1000)
        metrics   = evaluate(generated)
        print(metrics)
    """
    if isinstance(generated, torch.Tensor):
        generated = generated.cpu().numpy()

    metrics = {}
    metrics.update(covariance_error(Sigma, generated))
    metrics.update(wasserstein_error(generated, n_real=n_real))
    return metrics