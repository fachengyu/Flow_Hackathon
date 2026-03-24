"""
DiT-style transformer backbone for the MDLM workshop notebook.

Architecture:
  MDLMBackbone
  ├── vocab_embed     nn.Embedding(vocab_size, hidden_size)
  ├── sigma_map       TimestepEmbedder  (sinusoidal → MLP → cond_dim)
  ├── blocks          N × DDiTBlock     (bidirectional attention + AdaLN)
  └── output          AdaLN LayerNorm + linear projection → vocab_size

Key design choices:
  - **Learned absolute positional embeddings**: added to the token embeddings
    so the model knows which position each token occupies. Text is ordered, and
    position information is essential for learning coherent language patterns.
  - AdaLN (Adaptive Layer Norm): the timestep embedding modulates the layer
    norm inside each block, allowing the model to behave differently at
    different noise levels.
  - Standard bidirectional attention: every token attends to every other token
    (no causal masking), so unmasked tokens inform the predictions for masked ones.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """Layer norm without bias (bias adds parameters without clear benefit here)."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, [self.dim]) * self.weight


class TimestepEmbedder(nn.Module):
    """
    Embeds a scalar timestep t ∈ [0, 1] into a conditioning vector.

    Pipeline:
      t  →  sinusoidal(t, freq_dim)  →  Linear  →  SiLU  →  Linear  →  cond_dim
    """

    def __init__(self, cond_dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    @staticmethod
    def _sinusoidal(t: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] floats in [0, 1]. Returns [B, cond_dim]."""
        return self.mlp(self._sinusoidal(t, self.freq_dim))


class DDiTBlock(nn.Module):
    """
    Bidirectional transformer block with Adaptive Layer Norm (AdaLN).

    Standard pre-norm transformer block, extended with AdaLN:
      - Before attention:  h = norm1(x);  h = h * (1 + scale1) + shift1
      - After attention:   x = x + gate1 * attn_output
      - Before MLP:        h = norm2(x);  h = h * (1 + scale2) + shift2
      - After MLP:         x = x + gate2 * mlp_output

    The shift/scale/gate parameters (6 × hidden_size total) are predicted by
    a linear layer from the timestep conditioning vector c. They are initialized
    to zero so the block starts as an identity transformation.
    """

    def __init__(self, dim: int, n_heads: int, cond_dim: int,
                 mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dropout = dropout

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_ratio * dim, dim),
        )

        # AdaLN modulation: 6 parameters per hidden dim (shift/scale/gate × 2 sub-layers)
        # Zero-init so the block is an identity at the start of training
        self.adaLN_mod = nn.Linear(cond_dim, 6 * dim)
        nn.init.zeros_(self.adaLN_mod.weight)
        nn.init.zeros_(self.adaLN_mod.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x:         [B, L, D]  token representations
        c:         [B, cond_dim]  timestep conditioning
        attn_mask: [B, L] bool, True = valid token (optional)
                   When provided, PAD positions are masked out of attention keys.
        """
        mods = self.adaLN_mod(c).unsqueeze(1)              # [B, 1, 6*D]
        shift1, scale1, gate1, shift2, scale2, gate2 = mods.chunk(6, dim=-1)

        # Self-attention with AdaLN pre-modulation
        h = self.norm1(x) * (1 + scale1) + shift1
        B, L, D = h.shape
        qkv = self.attn_qkv(h).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                        # each [B, L, H, d_head]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # Build additive bias mask: 0 for valid keys, -inf for PAD keys.
        sdpa_mask = None
        if attn_mask is not None:
            # attn_mask: [B, L] bool (True = valid). Shape [B, 1, 1, L] for broadcasting over heads/queries.
            sdpa_mask = torch.zeros(attn_mask.shape[0], 1, 1, attn_mask.shape[1],
                                    device=q.device, dtype=q.dtype)
            sdpa_mask.masked_fill_(~attn_mask[:, None, None, :], float('-inf'))
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=sdpa_mask,
            dropout_p=self.dropout if self.training else 0.0
        )
        attn = attn.transpose(1, 2).reshape(B, L, D)
        x = x + gate1 * F.dropout(self.attn_out(attn), p=self.dropout, training=self.training)

        # MLP with AdaLN pre-modulation
        h = self.norm2(x) * (1 + scale2) + shift2
        x = x + gate2 * F.dropout(self.mlp(h), p=self.dropout, training=self.training)
        return x




# ---------------------------------------------------------------------------
# Full Backbone
# ---------------------------------------------------------------------------

class MDLMBackbone(nn.Module):
    """
    DiT-style transformer backbone for MDLM.

    Input:  (x_t: token ids [B, L],  t: timestep [B])
    Output: logits [B, L, vocab_size]

    The model predicts the original token x_0 at every masked position
    given the noisy input x_t and the noise level t.
    """

    def __init__(self, vocab_size: int, hidden_size: int, n_blocks: int,
                 n_heads: int, cond_dim: int, seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.vocab_embed = nn.Embedding(vocab_size, hidden_size)
        nn.init.kaiming_uniform_(self.vocab_embed.weight, a=math.sqrt(5))

        self.pos_embed = nn.Embedding(seq_len, hidden_size)

        self.sigma_map = TimestepEmbedder(cond_dim)
        self.blocks = nn.ModuleList([
            DDiTBlock(hidden_size, n_heads, cond_dim, dropout=dropout)
            for _ in range(n_blocks)
        ])

        self.norm_final = LayerNorm(hidden_size)
        # AdaLN for the final layer
        self.final_adaLN = nn.Linear(cond_dim, 2 * hidden_size)
        nn.init.zeros_(self.final_adaLN.weight)
        nn.init.zeros_(self.final_adaLN.bias)
        # Output projection — zero-init for stable training start
        self.output_proj = nn.Linear(hidden_size, vocab_size)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                seq_lens: torch.Tensor = None) -> torch.Tensor:
        """
        x:        [B, L]  int token ids (may contain MASK_ID or PAD_ID)
        t:        [B]     float timesteps in [0, 1]
        seq_lens: [B]     number of valid (non-PAD) tokens per sequence (optional).
                          When provided, PAD positions are excluded from attention keys.
        returns:  [B, L, vocab_size] raw logits
        """
        B, L = x.shape
        pos = torch.arange(L, device=x.device)
        h = self.vocab_embed(x) + self.pos_embed(pos)   # [B, L, D]
        c = F.silu(self.sigma_map(t))                    # [B, cond_dim]

        # Build boolean validity mask: True = valid token, False = PAD
        attn_mask = None
        if seq_lens is not None:
            attn_mask = torch.arange(L, device=x.device)[None, :] < seq_lens[:, None]  # [B, L]

        for block in self.blocks:
            h = block(h, c, attn_mask=attn_mask)

        # Final projection with AdaLN
        h = self.norm_final(h)
        shift, scale = self.final_adaLN(c).unsqueeze(1).chunk(2, dim=-1)
        h = h * (1 + scale) + shift
        return self.output_proj(h)                  # [B, L, vocab_size]
