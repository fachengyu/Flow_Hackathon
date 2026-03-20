# %% [markdown]
# # Masked Diffusion Models
#
# This notebook implements a **Masked Diffusion Language Model (MDLM)** — a
# generative model for discrete sequences based on a *masking* forward process.
#
# ## Diffusion vs. Flow Matching for Discrete Data
#
# Continuous diffusion models have a natural discrete counterpart: instead of
# adding Gaussian noise, we corrupt tokens by randomly replacing them with a
# special `[MASK]` token. The model learns to reverse this corruption.
#
# An alternative framing is **discrete flow matching**, which arrives at a very
# similar algorithm from a flow-matching perspective (Campbell et al., 2024).
# In practice the two approaches coincide for the absorbing-state (masking)
# process we use here, so we follow the diffusion convention.
#
# ## Overview
#
# | Step | What happens |
# |------|-------------|
# | **Forward process** $q(x_t \mid x_0)$ | Randomly mask tokens; more masking as $t \to 1$ |
# | **Reverse model** $p_\theta(x_0 \mid x_t)$ | Bidirectional transformer predicts original tokens |
# | **Training** | Maximize an ELBO — up-weight low-noise (easy) denoising steps |
# | **Sampling** | Start fully masked; iteratively unmask via the reverse posterior |
#
# We train on **text8** — a character-level dataset of English Wikipedia (~100M characters).
#
# **References**:
# - Austin et al., *Structured Denoising Diffusion Models in Discrete State-Spaces* (NeurIPS 2021)
# - Sahoo et al., *Simple and Effective Masked Diffusion Language Models* (NeurIPS 2024)
# - Campbell et al., *A Generative Model for Discrete Tokenized Data* (NeurIPS 2024) — [[pdf]](https://arxiv.org/pdf/2402.04997)

# %% [markdown]
# ## Setup

# %%
import sys
import os
import math

# Ensure the workshop/ directory is on the path (for mdlm_data, mdlm_model)
sys.path.insert(0, os.path.abspath('.'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from mdlm_data import (
    Text8Dataset, _download_text8, decode,
    VOCAB, VOCAB_SIZE, MASK_ID, NEG_INF,
    bits_per_char,
)
from mdlm_model import MDLMBackbone

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
print(f'Vocabulary size: {VOCAB_SIZE}  ([MASK]=0, a=1, ..., z=26, space=27)')

# %% [markdown]
# ## 1. Data
#
# text8 is a 100M-character slice of Wikipedia, lowercased to 27 characters
# (a–z and space). We chunk it into non-overlapping fixed-length sequences.
#
# The vocabulary has 28 tokens: [MASK] + 26 letters + space.

# %%
CACHE_DIR = '../artifacts/text8_standalone'
SEQ_LEN   = 64   # characters per sequence

splits     = _download_text8(CACHE_DIR)
train_ds   = Text8Dataset(splits['train'], SEQ_LEN)
val_ds     = Text8Dataset(splits['val'],   SEQ_LEN)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False,
                          num_workers=2, drop_last=False)

print(f'Train: {len(train_ds):,} sequences  |  Val: {len(val_ds):,} sequences')

# Build word vocabulary from training text (used for WER evaluation).
# A "word" is any space-delimited token in the raw training string.
word_vocab = frozenset(splits['train'].split())
print(f'Word vocabulary: {len(word_vocab):,} unique words')
print()
print('Sample sequences:')
for i in range(5):
    print(f'  [{i}]  "{decode(train_ds[i])}"')

# %% [markdown]
# ---
# ## 2. The Forward Process
#
# ### 2.1 Absorbing-state masking
#
# The forward process independently masks each token with probability $(1 - \alpha_t)$:
#
# $$q(x_t^i \mid x_0^i) = \alpha_t \cdot \delta[x_t^i = x_0^i] \;+\; (1 - \alpha_t) \cdot \delta[x_t^i = \texttt{[MASK]}]$$
#
# - With probability $\alpha_t$: keep the original token.
# - With probability $(1 - \alpha_t)$: replace it with $\texttt{[MASK]}$.
#
# Tokens mask independently — the joint distribution factorizes over positions.
#
# ### 2.2 Linear Noise Schedule
#
# We need $\alpha_t$ to smoothly decrease from 1 (clean) to 0 (fully masked) as $t: 0 \to 1$.
# We use a simple **linear schedule**:
#
# $$\alpha_t = 1 - (1 - \varepsilon) \cdot t, \qquad \varepsilon = 10^{-3}$$
#
# The small $\varepsilon$ keeps $\alpha_t > 0$ even at $t = 1$ for numerical stability.
# The derivative $d\alpha_t/dt = -(1 - \varepsilon)$ is constant.

# %%
class LinearSchedule(nn.Module):
    """
    Linear noise schedule:  alpha_t = 1 - (1 - eps) * t

    Returns (dalpha_t, alpha_t) for a batch of timesteps t in [0, 1].
    dalpha_t = d(alpha_t)/dt is used in the ELBO loss weight.
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, t: torch.Tensor):
        """t: [B] floats in [0, 1]. Returns (dalpha_t, alpha_t), each [B]."""
        # ── EXERCISE 1: Linear noise schedule ─────────────────────────────────
        # Implement the linear schedule:
        #
        #   alpha_t  = 1 - (1 - eps) * t
        #   dalpha_t = d(alpha_t)/dt = -(1 - eps)   (constant)
        #
        # Both outputs should be tensors of shape [B], matching the input t.
        # Use torch.full_like to create a constant tensor of the right shape.
        # ── END EXERCISE 1 ────────────────────────────────────────────────────
        t_scaled = (1 - self.eps) * t
        alpha_t  = 1.0 - t_scaled
        dalpha_t = torch.full_like(t, -(1 - self.eps))
        return dalpha_t, alpha_t


# Quick sanity check
noise = LinearSchedule()
t_test = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
dalpha, alpha = noise(t_test)
print('t       alpha_t   mask_prob')
print('-' * 32)
for t_, a_ in zip(t_test, alpha):
    print(f'  {t_:.2f}    {a_:.3f}     {1 - a_:.3f}')


# %% [markdown]
# ### 2.3 Sampling from the Forward Process: $q(x_t \mid x_0)$
#
# Given a clean sequence $x_0$ and noise level $\alpha_t$, we sample $x_t \sim q(\cdot \mid x_0)$ by
# independently masking each position:
#
# $$x_t^i = \begin{cases} \texttt{[MASK]} & \text{with probability } 1 - \alpha_t \\ x_0^i & \text{with probability } \alpha_t \end{cases}$$

# %%
def q_xt(x0: torch.Tensor, alpha_t: torch.Tensor) -> torch.Tensor:
    """
    Sample x_t from the forward process q(x_t | x_0).

    Args:
        x0:      [B, L]  clean token ids
        alpha_t: [B, 1]  keep probability per sample, broadcast over positions
    Returns:
        xt:      [B, L]  noisy sequence with some tokens replaced by MASK_ID
    """
    # ── EXERCISE 2: Forward process — corrupt x0 ──────────────────────────────
    # Sample x_t ~ q(x_t | x_0) by independently masking each token:
    #
    #   for each position i:
    #       draw u ~ Uniform[0, 1]
    #       x_t[i] = MASK_ID  if u < 1 - alpha_t
    #              = x_0[i]   otherwise
    #
    # Use torch.rand_like to draw per-token uniform noise, then torch.where
    # to select MASK_ID or the original token.  Note that alpha_t is shaped
    # [B, 1] so it broadcasts over the L dimension automatically.
    # ── END EXERCISE 2 ────────────────────────────────────────────────────────
    rand = torch.rand_like(x0, dtype=torch.float)  # independent uniform noise
    mask = rand < (1.0 - alpha_t)                  # True where we should mask
    return torch.where(mask, MASK_ID, x0)


# Demonstrate: corrupt the same sequence at increasing noise levels
noise   = LinearSchedule()
example = train_ds[42].unsqueeze(0)   # [1, L]
print(f'Original:  "{decode(example[0])}"')
print()

for t_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
    t      = torch.tensor([t_val])
    _, a   = noise(t)
    xt     = q_xt(example, a.unsqueeze(-1))
    frac   = (xt == MASK_ID).float().mean().item()
    print(f't={t_val:.1f} ({frac:.0%} masked):  "{decode(xt[0])}"')


# %% [markdown]
# ---
# ## 3. The Reverse Model $p_\theta(x_0 \mid x_t)$
#
# The forward process tells us how to corrupt data. To generate new sequences we
# need to *reverse* it: given a noisy sequence $x_t$, predict the original $x_0$.
# This is the distribution $p_\theta(x_0 \mid x_t)$ — a transformer with learnable
# parameters $\theta$.
#
# ### Connection to BERT
#
# This denoising task is exactly **masked language modelling**: the model sees a
# partially masked sequence and predicts the identity of the masked tokens from
# context. 
#
# ### Comparison to autoregressive models
#
# Autoregressive models (GPT-style) factorise the distribution left to right:
# each token is predicted from only the tokens before it, using a causally masked
# transformer. MDLM differs in two important ways:
#
# - **Bidirectional attention**: every position attends to every other position,
#   so predictions at masked locations benefit from the full available context.
# - **Order-agnostic generation**: tokens are revealed in a random order over
#   multiple steps rather than strictly left to right.
# - **Parallel decoding**: multiple tokens can be generated in the same step. However, 
#   they would be sampled from independent categorical distributions and unable to condition on each other. 


# %% [markdown]
# ---
# ## 4. Model Architecture
#
# We use a **DiT** (Diffusion Transformer) — a bidirectional transformer with one key
# modification: **Adaptive Layer Norm (AdaLN)**, which conditions every block on the
# timestep $t$.
#
# The backbone is fully defined in `mdlm_model.py`. Key properties:
#
# - **Full bidirectional attention**: every token attends to every other token,
#   so unmasked context can inform predictions at masked positions. Unlike
#   autoregressive models, there is no causal mask.
# - **AdaLN time conditioning**: the timestep $t$ is embedded and used to compute
#   per-channel shift, scale, and gate parameters inside every block and the
#   final layer. Initialized to zero (identity at training start).
#
# Architecture:
#
#   $x_t$ [B, L]
#     → Embedding [B, L, D]
#     → DDiTBlock × N  (bidirectional attn + AdaLN, conditioned on $t$)
#     → AdaLN LayerNorm
#     → Linear [B, L, vocab_size]

# %%
# Instantiate the backbone (defined in mdlm_model.py)
HIDDEN_SIZE = 128
N_BLOCKS    = 5
N_HEADS     = 8
COND_DIM    = 128
DROPOUT     = 0.1

backbone = MDLMBackbone(
    vocab_size=VOCAB_SIZE,
    hidden_size=HIDDEN_SIZE,
    n_blocks=N_BLOCKS,
    n_heads=N_HEADS,
    cond_dim=COND_DIM,
    seq_len=SEQ_LEN,
    dropout=DROPOUT,
)
n_params = sum(p.numel() for p in backbone.parameters())
print(f'MDLMBackbone parameters: {n_params:,}')

# Test forward pass: takes noisy token ids + timestep, returns logits
x_test = train_ds[:4]           # [4, 64]
t_test = torch.rand(4)          # [4] in [0, 1]
logits = backbone(x_test, t_test)
print(f'Input shape:   {tuple(x_test.shape)}')
print(f'Output shape:  {tuple(logits.shape)}   (batch, seq_len, vocab_size)')

# %% [markdown]
# ---
# ## 5. The Training Objective
#
# The model is trained by maximising a variational lower bound (ELBO) on the
# log-likelihood. We state the result here — see Sahoo et al. (2024) for the
# full derivation.
#
# For a continuous-time absorbing-state process the ELBO is:
#
# $$\mathcal{L}(\theta) = \mathbb{E}_{t \sim U[0,1]} \left[ \frac{-\dot\alpha_t}{1 - \alpha_t} \cdot \mathbb{E}_{x_t \sim q(\cdot|x_0)} \left[ -\log p_\theta(x_0 \mid x_t) \right] \right]$$
#
# With the linear schedule $\alpha_t = 1 - (1-\varepsilon)t$ the weight simplifies (note $\dot\alpha_t < 0$):
#
# $$\frac{-\dot\alpha_t}{1 - \alpha_t} = \frac{1-\varepsilon}{(1-\varepsilon)\,t} = \frac{1}{t}$$
#
# So the loss reduces to:
#
# $$\mathcal{L}(\theta) = \mathbb{E}_t\!\left[\frac{1}{t} \sum_{i:\, x_t^i = \texttt{[MASK]}} -\log p_\theta(x_0^i \mid x_t)\right]$$
#
# ### Interpretation: a weighted sum of MLM losses
#
# For a fixed $t$, the inner expectation is exactly a **masked language modelling
# (BERT-style) loss** — the model sees a partially masked sequence and must
# predict the original tokens at the masked positions.  The ELBO is therefore
# a *weighted average of MLM losses across all noise levels*.
#
# The $1/t$ weight has a natural interpretation: at noise level $t$, roughly
# $tL$ tokens are masked (in expectation).  Multiplying the sum of per-token
# losses by $1/t$ is therefore equivalent to **averaging over the masked
# positions**, keeping the contribution of each noise level on a consistent
# per-token scale regardless of how many tokens happen to be masked.
#
# **Implementation details**:
# 1. Only **masked positions** contribute to the loss.
# 2. The model **cannot output** $\texttt{[MASK]}$ — we set its logit to $-\infty$ before softmax.
# 3. Sample $t \in [\varepsilon, 1]$ to avoid the $1/t \to \infty$ singularity at $t = 0$.

# %%
def compute_loss(backbone, noise_schedule, x0, sampling_eps=1e-3):
    """
    ELBO loss for a batch of clean sequences x0.

    Steps:
      1. Sample t ~ Uniform[eps, 1] per sequence.
      2. Get (dalpha_t, alpha_t) from the noise schedule.
      3. Corrupt: xt = q_xt(x0, alpha_t).
      4. Forward pass: logits = backbone(xt, t).
      5. ELBO weight = |dalpha_t| / (1 - alpha_t) = 1/t.
      6. Loss = elbo_weight * CE at masked positions, normalized by mask count.

    Args:
        backbone:         MDLMBackbone
        noise_schedule:   LinearSchedule
        x0:               [B, L] clean token ids
        sampling_eps:     lower bound for t (avoids 1/t blowup near t=0)
    Returns:
        scalar loss
    """
    B = x0.shape[0]

    # ── EXERCISE 3: ELBO training loss ────────────────────────────────────────
    # Implement the MDLM ELBO loss.  Follow the six steps below.
    #
    # Step 1 — Sample t ~ Uniform[sampling_eps, 1] per sequence.
    #   (Avoid t=0 to prevent the 1/t weight from blowing up.)
    #
    # Step 2 — Get (dalpha_t, alpha_t) from noise_schedule(t).
    #   Unsqueeze both to shape [B, 1] so they broadcast over the L dimension.
    #
    # Step 3 — Corrupt: xt = q_xt(x0, alpha_t).
    #
    # Step 4 — Forward pass: logits = backbone(xt, t).   Shape: [B, L, vocab_size].
    #
    # Step 5 — Compute the ELBO-weighted cross-entropy at masked positions only:
    #   a. Zero out the [MASK] logit (model cannot output [MASK]):
    #          logits[:, :, MASK_ID] = NEG_INF
    #   b. log_probs = logits.log_softmax(-1)                  # [B, L, vocab_size]
    #   c. log_p_x0  = log_probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)  # [B, L]
    #   d. is_masked  = (xt == MASK_ID).float()                # [B, L]
    #   e. elbo_weight = |dalpha_t| / (1 - alpha_t)            # [B, 1]
    #                  = 1/t  (with the LinearSchedule schedule)
    #   f. per_token  = elbo_weight * (-log_p_x0) * is_masked  # [B, L]
    #
    # Step 6 — Normalize by number of masked tokens (clamp to ≥ 1):
    #   return per_token.sum() / is_masked.sum().clamp(min=1.0)
    # ── END EXERCISE 3 ────────────────────────────────────────────────────────

    # 1. Sample timesteps uniformly, avoiding t=0
    t = torch.rand(B, device=x0.device) * (1.0 - sampling_eps) + sampling_eps  # [B]

    # 2. Noise schedule values
    dalpha_t, alpha_t = noise_schedule(t)
    alpha_t  = alpha_t.unsqueeze(-1)    # [B, 1] — broadcast over sequence length
    dalpha_t = dalpha_t.unsqueeze(-1)   # [B, 1]

    # 3. Corrupt the input
    xt = q_xt(x0, alpha_t)              # [B, L]

    # 4. Model prediction
    logits = backbone(xt, t)            # [B, L, vocab_size]

    # 5a. The model can never predict [MASK] as an output
    logits[:, :, MASK_ID] = NEG_INF
    log_probs = logits.log_softmax(-1)  # [B, L, vocab_size]

    # 5b. Log-prob of the correct (clean) token at each position
    log_p_x0 = log_probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)  # [B, L]

    # 5c. Apply ELBO weight and mask: only masked positions contribute
    is_masked   = (xt == MASK_ID).float()               # [B, L]
    elbo_weight = (-dalpha_t) / (1.0 - alpha_t)         # [B, 1] = 1/t with the linear schedule
    per_token   = elbo_weight * (-log_p_x0) * is_masked  # [B, L]

    # 6. Normalize by number of masked tokens in this batch
    num_masked = is_masked.sum().clamp(min=1.0)
    return per_token.sum() / num_masked


# Quick check: compute loss on an untrained model
noise   = LinearSchedule()
x_batch = train_ds[:64]
loss    = compute_loss(backbone, noise, x_batch)
print(f'Untrained model loss: {loss.item():.3f} nats  ({bits_per_char(loss.item()):.3f} bpd)')

# %% [markdown]
# ---
# ## 6. The MDLM Class
#
# We now wrap the backbone, noise schedule, forward process ($q\_xt$), and ELBO
# loss into a single `nn.Module`. We will add the `sample` method later, once
# we have covered the reverse-process math.

# %%
class MDLM(nn.Module):
    """
    Masked Diffusion Language Model.

    Bundles:
      - MDLMBackbone    (the transformer)
      - LinearSchedule  (the noise schedule)
      - q_xt            (forward/corruption process)
      - loss            (ELBO training objective)
    """

    def __init__(self, vocab_size, hidden_size, n_blocks, n_heads,
                 cond_dim, seq_len, dropout=0.1):
        super().__init__()
        self.seq_len  = seq_len
        self.backbone = MDLMBackbone(vocab_size, hidden_size, n_blocks,
                                     n_heads, cond_dim, seq_len, dropout)
        self.noise    = LinearSchedule()

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def q_xt(self, x0: torch.Tensor, alpha_t: torch.Tensor) -> torch.Tensor:
        """Corrupt x0 by masking each token independently with prob (1 - alpha_t)."""
        rand = torch.rand_like(x0, dtype=torch.float)
        return torch.where(rand < (1.0 - alpha_t), MASK_ID, x0)

    # ------------------------------------------------------------------
    # Training objective
    # ------------------------------------------------------------------

    def loss(self, x0: torch.Tensor, sampling_eps: float = 1e-3) -> torch.Tensor:
        """
        ELBO loss = E_t [ (1/t) * CE(backbone(x_t, t), x_0) ] at masked positions.
        See compute_loss() above for a detailed step-by-step walkthrough.
        """
        # ── EXERCISE 4: Wire up the ELBO loss ─────────────────────────────────
        # Call compute_loss() with the correct arguments.
        # You have access to self.backbone and self.noise inside the class.
        # ── END EXERCISE 4 ────────────────────────────────────────────────────
        return compute_loss(self.backbone, self.noise, x0, sampling_eps)


# %%
# Instantiate the model
model = MDLM(
    vocab_size=VOCAB_SIZE,
    hidden_size=HIDDEN_SIZE,
    n_blocks=N_BLOCKS,
    n_heads=N_HEADS,
    cond_dim=COND_DIM,
    seq_len=SEQ_LEN,
    dropout=DROPOUT,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {n_params:,}')

loss_init = model.loss(train_ds[:64].to(device))
print(f'Initial loss: {loss_init.item():.3f} nats  ({bits_per_char(loss_init.item()):.3f} bpd)')

# %% [markdown]
# ---
# ## 7. Training
#
# We use **AdamW** with a short linear warmup. The main risk during training is
# gradient spikes from the $1/t$ ELBO weight when $t$ is small — gradient clipping
# (max norm 1.0) keeps things stable.

# %%
def train_epoch(model, loader, optimizer, device, warmup_steps, step, base_lr):
    """Run one epoch. Returns (avg_loss_nats, updated_step)."""
    model.train()
    total_loss, n = 0.0, 0
    for x in loader:
        x = x.to(device)
        # Linear warmup
        lr_scale = min(1.0, step / max(warmup_steps, 1))
        for pg in optimizer.param_groups:
            pg['lr'] = base_lr * lr_scale

        optimizer.zero_grad()
        loss = model.loss(x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n    += 1
        step += 1
    return total_loss / max(n, 1), step


@torch.no_grad()
def evaluate(model, loader, device, max_batches=50):
    """Estimate validation loss on a subset of the val set."""
    model.eval()
    total_loss, n = 0.0, 0
    for i, x in enumerate(loader):
        if i >= max_batches:
            break
        total_loss += model.loss(x.to(device)).item()
        n += 1
    return total_loss / max(n, 1)

# %%
BASE_LR      = 3e-4
WARMUP_STEPS = 500
N_EPOCHS     = 20
CKPT_PATH    = '../artifacts/mdlm_workshop_best.pt'

os.makedirs(os.path.dirname(os.path.abspath(CKPT_PATH)), exist_ok=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.01)
history   = {'train': [], 'val': []}
best_val  = float('inf')
step      = 0

for epoch in range(1, N_EPOCHS + 1):
    train_loss, step = train_epoch(model, train_loader, optimizer,
                                   device, WARMUP_STEPS, step, BASE_LR)
    val_loss = evaluate(model, val_loader, device)

    history['train'].append(train_loss)
    history['val'].append(val_loss)

    print(f'Epoch {epoch:3d}/{N_EPOCHS}  |  '
          f'train {train_loss:.4f} ({bits_per_char(train_loss):.3f} bpd)  |  '
          f'val {val_loss:.4f} ({bits_per_char(val_loss):.3f} bpd)')

    if val_loss < best_val:
        best_val = val_loss
        torch.save({'model': model.state_dict(), 'epoch': epoch,
                    'val_bpd': bits_per_char(val_loss)}, CKPT_PATH)
        print(f'             checkpoint saved  (val bpd = {bits_per_char(val_loss):.3f})')

print(f'\nBest checkpoint saved to: {CKPT_PATH}')

# %%
# Plot the training curve
fig, ax = plt.subplots(figsize=(8, 4))
epochs = list(range(1, len(history['train']) + 1))
ax.plot(epochs, [bits_per_char(l) for l in history['train']], label='Train', color='steelblue')
ax.plot(epochs, [bits_per_char(l) for l in history['val']],   label='Val',   color='crimson')
ax.set_xlabel('Epoch')
ax.set_ylabel('Bits per character (BPD)')
ax.set_title('MDLM Training Curve on text8')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
print(f'Best val BPD: {bits_per_char(best_val):.3f}')

# %% [markdown]
# ---
# ## 8. Sampling: The Reverse Process
#
# Sampling reverses the forward process. We start fully masked and move
# through noise levels $t = 1 \to 0$ in `num_steps` equal increments.
#
# ### Key idea
#
# The model is trained to predict $x_0$ directly from $x_t$. But during sampling
# we dont want to jump all the way to $x_0$ from $x_t$ (this would generate indpendent tokens).
# Instead, at each step we take a *small* step from $t$ to $s = t - \Delta t$,
# producing $x_s$ which is still partially masked. The math below is working out what we should do
# with a prediction of $x_0$ to get a sample $x_s$ -- turns out we should randomly unmask a subset of the predictions.
#
# ### The posterior
#
# Given $x_t$ and $x_0$, the forward process tells us exactly what $x_s$ looks like —
# a masked token at time $t$ is either revealed (with probability $\alpha_s/\alpha_t$)
# or stays masked (with probability $1 - \alpha_s/\alpha_t$):
#
# $$q(x_s \mid x_t = \texttt{[MASK]},\, x_0) = \frac{\alpha_s}{\alpha_t} \cdot \delta[x_s = x_0] + \left(1 - \frac{\alpha_s}{\alpha_t}\right) \cdot \delta[x_s = \texttt{[MASK]}]$$
#
# Since we don't know $x_0$, we replace it with the model's prediction
# $p_\theta(x_0 \mid x_t)$:
#
# $$p(x_s = v) = p_\theta(x_0{=}v \mid x_t) \cdot (1 - m), \qquad p(x_s = \texttt{[MASK]}) = m, \qquad m = \frac{1 - \alpha_s}{1 - \alpha_t}$$
#
# The factor $m$ is the fraction of the remaining masking budget consumed by this
# step: when $\Delta t$ is small, most masked tokens stay masked and only a few
# are revealed, and future steps can now condition on these new tokens.
#
# **Gumbel-max?**
# Adding $\mathrm{Gumbel}(0,1)$ noise to log-probabilities before argmax is equivalent to
# sampling from the categorical.

# %%
@torch.no_grad()
def _sample(self, n_samples: int, num_steps: int, device: torch.device,
            return_trajectory: bool = False):
    """
    Generate sequences by simulating the MDLM reverse process.

    Args:
        n_samples:          number of sequences to generate
        num_steps:          reverse-process discretisation steps
        device:             torch device
        return_trajectory:  if True, also return list of intermediate states
    """
    x = torch.full((n_samples, self.seq_len), MASK_ID, dtype=torch.long, device=device)
    trajectory = [x.clone()] if return_trajectory else None

    for step in range(num_steps):
        # ── EXERCISE 5: One reverse-process step ──────────────────────────
        # Implement the body of the loop.  Each iteration should:
        #
        # (a) Compute current time t and next time s.
        #       t_val = 1.0 - step / num_steps
        #       s_val = 1.0 - (step + 1) / num_steps
        #     Clamp t to ≥ 1e-4 (model was not trained at t=0).
        #     Expand to shape [B] with torch.full.
        #
        # (b) Get alpha values from the noise schedule:
        #       _, alpha_t = self.noise(t)   # [B]
        #       _, alpha_s = self.noise(s)   # [B]
        #     Compute the "stay masked" probability:
        #       move_t    = 1 - alpha_t           # [B]
        #       move_s    = 1 - alpha_s           # [B]
        #       mask_prob = (move_s / move_t.clamp(min=1e-8))[:, None, None]  # [B, 1, 1]
        #
        # (c) Run the backbone and get p_θ(x_0 | x_t):
        #       logits = self.backbone(x, t)     # [B, L, vocab_size]
        #       logits[:, :, MASK_ID] = NEG_INF  # model never outputs [MASK]
        #       p_x0 = logits.softmax(dim=-1)    # [B, L, vocab_size]
        #
        # (d) Build posterior q(x_s | x_t, p_x0):
        #       q_xs = p_x0 * (1.0 - mask_prob)          # [B, L, vocab_size]
        #       q_xs[:, :, MASK_ID] = mask_prob[:, :, 0]  # [B, L]
        #
        # (e) Gumbel-max sample from q_xs:
        #       log_q  = q_xs.log().clamp(min=-1e9)
        #       gumbel = -log(-log(Uniform(0,1) + eps) + eps)
        #       x_new  = (log_q + gumbel).argmax(dim=-1)  # [B, L]
        #
        # (f) Keep already-revealed tokens (absorbing state):
        #       x = torch.where(x == MASK_ID, x_new, x)
        # ── END EXERCISE 5 ────────────────────────────────────────────────

        # (a) Current time t → next time s
        t_val = 1.0 - step / num_steps
        s_val = 1.0 - (step + 1) / num_steps
        t = torch.full((n_samples,), max(t_val, 1e-4), device=device)
        s = torch.full((n_samples,), max(s_val, 0.0),  device=device)

        # (b) Noise schedule and "stay masked" probability
        _, alpha_t = self.noise(t)                                       # [B]
        _, alpha_s = self.noise(s)                                       # [B]
        move_t    = 1.0 - alpha_t                                        # [B]
        move_s    = 1.0 - alpha_s                                        # [B]
        mask_prob = (move_s / move_t.clamp(min=1e-8))[:, None, None]      # [B, 1, 1]

        # (c) Model: p_θ(x_0 | x_t)
        logits = self.backbone(x, t)                                     # [B, L, V]
        logits[:, :, MASK_ID] = NEG_INF
        p_x0 = logits.softmax(dim=-1)                                    # [B, L, V]

        # (d) Posterior q(x_s | x_t, p_x0)
        q_xs = p_x0 * (1.0 - mask_prob)                                  # [B, L, V]
        q_xs[:, :, MASK_ID] = mask_prob[:, :, 0]                         # [B, L]

        # (e) Gumbel-max sample
        log_q  = q_xs.log().clamp(min=-1e9)
        gumbel = -torch.log(-torch.log(torch.rand_like(log_q) + 1e-10) + 1e-10)
        x_new  = (log_q + gumbel).argmax(dim=-1)                        # [B, L]

        # (f) Absorbing state: keep already-revealed tokens
        x = torch.where(x == MASK_ID, x_new, x)

        if return_trajectory:
            trajectory.append(x.clone())

    return (x, trajectory) if return_trajectory else x


MDLM.sample = _sample

# %% [markdown]
# ---
# ## 9. Results
#
# If you are resuming from a saved checkpoint (e.g. after restarting the kernel),
# run the cell below to reload the best model weights before generating samples.

# %%
# Optional: load the best checkpoint saved during training.
# Skip this cell if the model is already trained in the current session.
if os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt['model'])
    print(f'Loaded checkpoint from epoch {ckpt["epoch"]}  '
          f'(val bpd = {ckpt["val_bpd"]:.3f})')
else:
    print(f'No checkpoint found at {CKPT_PATH} — using current model weights.')

# %%
# Visualize the unmasking trajectory for a single generated sequence
model.eval()
NUM_STEPS = SEQ_LEN   # one token revealed per step

_, trajectory = model.sample(n_samples=1, num_steps=NUM_STEPS,
                              device=device, return_trajectory=True)

snap_steps = [0, NUM_STEPS // 8, NUM_STEPS // 4, NUM_STEPS // 2,
              3 * NUM_STEPS // 4, NUM_STEPS]
print('Unmasking trajectory (one generated sequence):')
print(f'  {"Step":>4}   t≈    masked   Sequence')
print('  ' + '-' * 72)
for s in snap_steps:
    t_approx  = 1.0 - s / NUM_STEPS
    seq       = decode(trajectory[s][0])
    mask_frac = seq.count('_') / len(seq)
    print(f'  {s:>4}  {t_approx:.2f}   {mask_frac:.0%}      "{seq}"')

# %%
# Generate a batch of samples and compare to real val sequences
N_DISPLAY = 8
samples   = model.sample(n_samples=N_DISPLAY, num_steps=SEQ_LEN, device=device)

print('Generated samples:')
print('-' * 70)
for i, s in enumerate(samples):
    print(f'  [{i}]  "{decode(s)}"')

print()
print('Reference (val) sequences:')
print('-' * 70)
for i in range(N_DISPLAY):
    print(f'  [{i}]  "{decode(val_ds[i])}"')

# %% [markdown]
# ---
# ## 10. Evaluation
#
# ### Bits per Character (BPD)
# The validation ELBO loss in bits per character (upper bound on true NLL). Lower is better.
#
# ### Word Error Rate (WER)
# Fraction of characters in generated text that belong to words **not** in the
# training vocabulary. Range $[0, 1]$: $0$ = all generated words are real words,
# $1$ = no generated words appear in the training data.
# The vocabulary is built from the raw training split (built earlier in the notebook).
#
# ### Num-steps trade-off
# More unmasking steps → better sample quality, but slower generation.
# We sweep over num_steps to see the trade-off.

# %%
# BPD on a larger portion of the val set
print('Computing validation BPD ...')
full_val_loss = evaluate(model, val_loader, device, max_batches=200)
val_bpd       = bits_per_char(full_val_loss)
print(f'Validation BPD: {val_bpd:.4f}')

# %%
# WER: fraction of characters in words not found in the training vocabulary

def word_error_rate(samples, vocab):
    """Fraction of characters belonging to words absent from the training vocab."""
    total_chars = invalid_chars = 0
    for text in samples:
        for word in text.split():
            total_chars += len(word)
            if word not in vocab:
                invalid_chars += len(word)
    return invalid_chars / max(total_chars, 1)

print('Computing WER on 100 generated sequences ...')
N_WER    = 100
gen_seqs = model.sample(n_samples=N_WER, num_steps=SEQ_LEN, device=device)
hyps     = [decode(s) for s in gen_seqs]
wer      = word_error_rate(hyps, word_vocab)
print(f'WER (chars in unknown words): {wer:.3f}')
print()
print('Example generated sequences:')
for h in hyps[:5]:
    print(f'  "{h}"')

# %%
# Sweep num_steps and measure WER — more steps → better quality but slower
print('Sweeping num_steps ...')

step_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256]  # up to 4 * SEQ_LEN
wers        = []

for ns in step_counts:
    seqs = model.sample(n_samples=64, num_steps=ns, device=device)
    h    = [decode(s) for s in seqs]
    wers.append(word_error_rate(h, word_vocab))
    print(f'  num_steps={ns:3d}   WER={wers[-1]:.3f}')

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.semilogx(step_counts, wers, 'o-', color='steelblue', ms=6)
ax.set_xlabel('Number of sampling steps')
ax.set_ylabel('WER (generated vs. val)')
ax.set_title('Sample Quality vs. Number of Unmasking Steps')
ax.set_xticks(step_counts)
ax.set_xticklabels(step_counts)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
