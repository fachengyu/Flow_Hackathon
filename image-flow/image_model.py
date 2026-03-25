"""
U-Net velocity field for conditional flow matching on images.

Architecture (following DDPM / ADM conventions):
    - Sinusoidal time embedding -> MLP -> time vector
    - Optional class embedding (for conditional generation / CFG)
    - Encoder: ResBlocks with downsampling, collecting skip connections
    - Decoder: ResBlocks with upsampling, consuming skip connections
    - Each ResBlock uses FiLM conditioning (scale + shift from embedding)

OT-CFM loss:
    x_t = (1 - t) * x_0 + t * x_1
    target = x_1 - x_0
    L = || v_theta(x_t, t) - target ||^2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Time embedding ───────────────────────────────────────────────────────────

def sincos_embed(t: torch.Tensor, dim: int, base: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding for scalar timesteps.  (B,) -> (B, dim)."""
    half = dim // 2
    k = torch.arange(half, device=t.device, dtype=t.dtype)
    w = base ** (-2 * k / dim)
    angles = t[:, None] * w[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class TimeEmbedding(nn.Module):
    """Sinusoidal time -> MLP -> embedding vector."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) floats in [0, 1]. Returns (B, embed_dim)."""
        h = sincos_embed(t, self.embed_dim)
        return self.mlp(h)


# ─── Building blocks ──────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """
    Residual conv block with FiLM conditioning (scale + shift from embedding).

    x -> GroupNorm -> SiLU -> Conv -> GroupNorm -> FiLM -> SiLU -> Conv -> + skip
    """

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        # FiLM: project embedding to scale and shift
        self.emb_proj = nn.Linear(emb_dim, out_ch * 2)

        # skip projection if channels change
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # FiLM conditioning: scale and shift after second norm
        scale_shift = self.emb_proj(emb)[:, :, None, None]
        scale, shift = scale_shift.chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift

        h = self.act(h)
        h = self.conv2(h)
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ─── U-Net ────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    Small U-Net for velocity prediction v_theta(x_t, t).

    Default config targets 28x28 MNIST with 2 downsample levels (28 -> 14 -> 7).
    Uses channel_mults=(1, 2) by default for ~3.5M params.

    Optional class conditioning for classifier-free guidance:
        - If num_classes > 0, a learnable class embedding is added to the
          time embedding. Class label `num_classes` is the null/unconditional class.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2),
        num_res_blocks: int = 2,
        embed_dim: int = 128,
        num_classes: int = 0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.channel_mults = channel_mults
        self.num_res_blocks = num_res_blocks

        # ── Embeddings ──
        self.time_embed = TimeEmbedding(embed_dim)
        if num_classes > 0:
            self.class_embed = nn.Embedding(num_classes + 1, embed_dim)

        # ── Encoder ──
        # Following DDPM/ADM: store skip connections after every ResBlock
        # and after every Downsample. The input_conv output is also a skip.
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        skip_channels = [base_channels]  # track all skip connection channel sizes
        ch = base_channels

        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(ResBlock(ch, out_ch, embed_dim))
                ch = out_ch
                skip_channels.append(ch)
            if level < len(channel_mults) - 1:
                self.downsamples.append(Downsample(ch))
                skip_channels.append(ch)

        # ── Bottleneck ──
        self.mid_block1 = ResBlock(ch, ch, embed_dim)
        self.mid_block2 = ResBlock(ch, ch, embed_dim)

        # ── Decoder ──
        # Each level has (num_res_blocks + 1) blocks to consume all skips,
        # plus an Upsample between levels (except the last).
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for level, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            n_blocks = num_res_blocks + (1 if level < len(channel_mults) - 1 else 0)
            for _ in range(n_blocks):
                skip_ch = skip_channels.pop()
                self.decoder_blocks.append(ResBlock(ch + skip_ch, out_ch, embed_dim))
                ch = out_ch
            if level < len(channel_mults) - 1:
                self.upsamples.append(Upsample(ch))

        # consume the input_conv skip
        skip_ch = skip_channels.pop()
        self.final_res = ResBlock(ch + skip_ch, base_channels, embed_dim)
        ch = base_channels
        assert len(skip_channels) == 0, f"Skip mismatch: {len(skip_channels)} remaining"

        # ── Output ──
        self.out_norm = nn.GroupNorm(min(32, ch), ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        class_label: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) noisy image x_t
            t: (B,) timestep in [0, 1]
            class_label: (B,) integer class labels (optional)
        Returns:
            (B, C, H, W) predicted velocity v_theta(x_t, t)
        """
        # conditioning embedding
        emb = self.time_embed(t)
        if self.num_classes > 0 and class_label is not None:
            emb = emb + self.class_embed(class_label)

        # encoder
        h = self.input_conv(x)
        skips = [h]

        block_idx = 0
        down_idx = 0
        for level in range(len(self.channel_mults)):
            for _ in range(self.num_res_blocks):
                h = self.encoder_blocks[block_idx](h, emb)
                skips.append(h)
                block_idx += 1
            if level < len(self.channel_mults) - 1:
                h = self.downsamples[down_idx](h)
                skips.append(h)
                down_idx += 1

        # bottleneck
        h = self.mid_block1(h, emb)
        h = self.mid_block2(h, emb)

        # decoder
        block_idx = 0
        up_idx = 0
        for level in range(len(self.channel_mults)):
            n_blocks = self.num_res_blocks + (1 if level < len(self.channel_mults) - 1 else 0)
            for _ in range(n_blocks):
                h = torch.cat([h, skips.pop()], dim=1)
                h = self.decoder_blocks[block_idx](h, emb)
                block_idx += 1
            if level < len(self.channel_mults) - 1:
                h = self.upsamples[up_idx](h)
                up_idx += 1

        # final block consumes input_conv skip
        h = torch.cat([h, skips.pop()], dim=1)
        h = self.final_res(h, emb)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)


# ─── CFM loss and sampling ────────────────────────────────────────────────────

def cfm_loss(
    model: UNet,
    x1: torch.Tensor,
    class_label: torch.Tensor = None,
    p_uncond: float = 0.0,
) -> torch.Tensor:
    """
    OT conditional flow matching loss for images.

    Args:
        model: UNet velocity field
        x1: (B, C, H, W) real images in [0, 1]
        class_label: (B,) integer class labels (optional)
        p_uncond: probability of dropping class label (for CFG training)
    """
    B = x1.shape[0]
    device = x1.device

    # sample noise and timesteps
    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=device)

    # OT interpolation: x_t = (1 - t) * x_0 + t * x_1
    t_expand = t[:, None, None, None]
    x_t = (1 - t_expand) * x0 + t_expand * x1

    # target velocity: x_1 - x_0
    target = x1 - x0

    # randomly drop class labels for CFG training
    if class_label is not None and p_uncond > 0:
        drop_mask = torch.rand(B, device=device) < p_uncond
        class_label = class_label.clone()
        class_label[drop_mask] = model.num_classes  # null class

    pred = model(x_t, t, class_label)
    return ((pred - target) ** 2).mean()


@torch.no_grad()
def sample(
    model: UNet,
    n_samples: int,
    n_steps: int = 50,
    device: str = "cpu",
    class_label: torch.Tensor = None,
    guidance_scale: float = 0.0,
    return_trajectory: bool = False,
    img_channels: int = 1,
    img_size: int = 28,
) -> torch.Tensor:
    """
    Euler integration from x_0 ~ N(0, I) to x_1 (generated images).

    Args:
        model: trained UNet
        n_samples: number of images to generate
        n_steps: number of Euler steps
        device: torch device
        class_label: (n_samples,) class labels for conditional generation
        guidance_scale: CFG weight w. v = v_uncond + w * (v_cond - v_uncond)
        return_trajectory: if True, return list of snapshots at each step
        img_channels: number of image channels
        img_size: spatial size of images

    Returns:
        (n_samples, C, H, W) generated images, or list of snapshots if requested
    """
    model.eval()
    x = torch.randn(n_samples, img_channels, img_size, img_size, device=device)
    dt = 1.0 / n_steps

    trajectory = [x.clone()] if return_trajectory else None

    null_label = None
    if class_label is not None and guidance_scale > 0:
        null_label = torch.full_like(class_label, model.num_classes)

    for i in range(n_steps):
        t = torch.full((n_samples,), i / n_steps, device=device)

        if class_label is not None and guidance_scale > 0:
            # classifier-free guidance: two forward passes
            v_cond = model(x, t, class_label)
            v_uncond = model(x, t, null_label)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            v = model(x, t, class_label)

        x = x + v * dt

        if return_trajectory:
            trajectory.append(x.clone())

    if return_trajectory:
        return trajectory
    return x


# ─── Reflow utilities ─────────────────────────────────────────────────────────

@torch.no_grad()
def generate_reflow_pairs(
    model: UNet,
    n_pairs: int,
    n_steps: int = 50,
    device: str = "cpu",
    batch_size: int = 256,
    img_channels: int = 1,
    img_size: int = 28,
) -> tuple:
    """
    Generate (x_0, x_1) pairs for reflow by running the trained model forward.

    For each noise sample x_0 ~ N(0, I), integrate the ODE to get x_1.
    Returns (noise_samples, generated_samples) each of shape (n_pairs, C, H, W).
    """
    model.eval()
    all_x0 = []
    all_x1 = []

    for start in range(0, n_pairs, batch_size):
        B = min(batch_size, n_pairs - start)
        x0 = torch.randn(B, img_channels, img_size, img_size, device=device)
        x = x0.clone()
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t = torch.full((B,), i / n_steps, device=device)
            x = x + model(x, t) * dt

        all_x0.append(x0.cpu())
        all_x1.append(x.cpu())

    return torch.cat(all_x0, dim=0), torch.cat(all_x1, dim=0)


def trajectory_straightness(
    model: UNet,
    n_samples: int = 64,
    n_steps: int = 50,
    device: str = "cpu",
    img_channels: int = 1,
    img_size: int = 28,
) -> float:
    """
    Measure trajectory straightness: displacement / path_length.

    Straightness = ||x_1 - x_0|| / sum_i ||x_{t+dt} - x_t||

    Returns a value in (0, 1]. Perfectly straight trajectories score 1.0.
    """
    model.eval()
    x = torch.randn(n_samples, img_channels, img_size, img_size, device=device)
    x_start = x.clone()
    dt = 1.0 / n_steps
    path_length = torch.zeros(n_samples, device=device)

    with torch.no_grad():
        for i in range(n_steps):
            t = torch.full((n_samples,), i / n_steps, device=device)
            v = model(x, t)
            step = v * dt
            path_length += step.flatten(1).norm(dim=1)
            x = x + step

    displacement = (x - x_start).flatten(1).norm(dim=1)
    straightness = (displacement / path_length.clamp(min=1e-8)).mean().item()
    return straightness
