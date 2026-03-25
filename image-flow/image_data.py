"""
MNIST dataset utilities for image flow matching.

Provides train/test DataLoaders that return images normalized to [0, 1]
and integer class labels in {0, ..., 9}.
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

IMG_CHANNELS = 1
IMG_SIZE = 28
NUM_CLASSES = 10


def get_mnist_loaders(batch_size: int = 128, root: str = "./data"):
    """
    Download MNIST and return (train_loader, test_loader).

    Images are float32 in [0, 1], shape (1, 28, 28).
    Labels are long tensors in {0, ..., 9}.
    """
    transform = T.Compose([T.ToTensor()])

    train_ds = torchvision.datasets.MNIST(
        root=root, train=True, download=True, transform=transform,
    )
    test_ds = torchvision.datasets.MNIST(
        root=root, train=False, download=True, transform=transform,
    )

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
