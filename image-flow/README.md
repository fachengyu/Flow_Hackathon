# Image Flow Matching

Hands-on tutorial for **continuous flow matching** applied to image generation, progressing from unconditional MNIST generation to class-conditional ImageNet sampling with pretrained models.

## Overview

This module teaches flow matching for images through a five-part walkthrough:

1. **Unconditional Flow Matching on MNIST** — Train a U-Net velocity field using OT conditional flow matching (OT-CFM), sample via Euler integration, and explore how the number of function evaluations (NFE) affects quality.
2. **Class-Conditional Generation with Classifier-Free Guidance (CFG)** — Add class conditioning to the U-Net, train with random label dropout, and generate specific digits with tunable guidance scale.
3. **Rectified Flow (Reflow)** — Straighten ODE trajectories by training on coupled (noise, image) pairs, enabling high-quality generation with fewer sampling steps.
4. **Advanced Sampling Techniques** — Heun's method (2nd-order ODE solver), spherical noise-space interpolation (slerp), and SDEdit-style image editing.
5. **Pretrained Flow Matching Models** — Load pretrained CIFAR-10 and ImageNet (DiT) models via Hugging Face Diffusers for higher-resolution generation.

## Files

| File | Description |
|------|-------------|
| `image_model.py` | U-Net architecture with FiLM conditioning, OT-CFM loss, Euler sampler, and reflow utilities |
| `image_data.py` | MNIST DataLoader and image grid visualization helpers |
| `image_flow_walkthrough_exercise.ipynb` | Exercise version with code sections to fill in |

## Key Concepts

- **OT Conditional Flow Matching**: Learn a velocity field `v_theta` that transports noise `x_0 ~ N(0, I)` to data `x_1` along the optimal-transport interpolation `x_t = (1-t) * x_0 + t * x_1`.
- **FiLM Conditioning**: The U-Net uses Feature-wise Linear Modulation (scale + shift) to inject timestep and class embeddings into residual blocks.
- **Classifier-Free Guidance**: During training, class labels are randomly dropped with probability `p_uncond`. At inference, the velocity is steered via `v = v_uncond + w * (v_cond - v_uncond)`.
- **Reflow**: Re-couple noise and generated samples to train straighter trajectories, measured by `displacement / path_length`.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- diffusers, accelerate (Part 5 only)

## Getting Started

- **`image_flow_walkthrough_exercise.ipynb`**

Training the MNIST models takes approximately 5-8 minutes per model on a T4 GPU.

## Model Architecture

The U-Net (~3.5M parameters) follows DDPM/ADM conventions:

- Sinusoidal time embedding projected through an MLP
- Encoder with ResBlocks + downsampling (28 -> 14 -> 7 for MNIST)
- Decoder with ResBlocks + upsampling, consuming skip connections
- Optional learnable class embedding (added to time embedding)
