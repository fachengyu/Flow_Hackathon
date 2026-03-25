"""
MNIST dataset utilities for image flow matching.

Provides train/test DataLoaders that return images normalized to [0, 1]
and integer class labels in {0, ..., 9}.
"""

import ssl
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

IMG_CHANNELS = 1
IMG_SIZE = 28
NUM_CLASSES = 10


def get_mnist_loaders(batch_size: int = 128, root: str | None = None):
    """
    Download MNIST and return (train_loader, test_loader).

    Images are float32 in [0, 1], shape (1, 28, 28).
    Labels are long tensors in {0, ..., 9}.
    """
    transform = T.Compose([T.ToTensor()])

    # Default to the module-local data dir so notebooks work from any cwd.
    data_root = Path(root).expanduser() if root is not None else Path(__file__).resolve().parent / "data"

    try:
        # Prefer existing local files and avoid network calls when possible.
        train_ds = torchvision.datasets.MNIST(
            root=str(data_root), train=True, download=False, transform=transform,
        )
        test_ds = torchvision.datasets.MNIST(
            root=str(data_root), train=False, download=False, transform=transform,
        )
    except RuntimeError:
        try:
            train_ds = torchvision.datasets.MNIST(
                root=str(data_root), train=True, download=True, transform=transform,
            )
            test_ds = torchvision.datasets.MNIST(
                root=str(data_root), train=False, download=True, transform=transform,
            )
        except RuntimeError as err:
            # Some cluster images miss CA roots; fallback to unverified SSL only for this download.
            if "CERTIFICATE_VERIFY_FAILED" not in str(err):
                raise

            original_https_context = ssl._create_default_https_context
            ssl._create_default_https_context = ssl._create_unverified_context
            try:
                train_ds = torchvision.datasets.MNIST(
                    root=str(data_root), train=True, download=True, transform=transform,
                )
                test_ds = torchvision.datasets.MNIST(
                    root=str(data_root), train=False, download=True, transform=transform,
                )
            finally:
                ssl._create_default_https_context = original_https_context

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    return train_loader, test_loader


def show_samples(images, nrow=8, title=None):
    """Display a grid of images. images: (N, 1, 28, 28) tensor in [0, 1]."""
    import matplotlib.pyplot as plt
    grid = torchvision.utils.make_grid(images.clamp(0, 1), nrow=nrow, padding=1)
    plt.figure(figsize=(nrow * 1.2, (len(images) // nrow + 1) * 1.2))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()
