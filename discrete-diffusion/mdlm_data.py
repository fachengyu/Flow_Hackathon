"""
Data utilities for the MDLM workshop notebook.
Handles text8 download, tokenization, and evaluation metrics.
"""

import math
import os
import urllib.request
import zipfile
from typing import List

import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
# [MASK] + 26 letters + space = 28 tokens total
VOCAB: dict = {
    '[MASK]': 0,
    **{ch: i + 1 for i, ch in enumerate('abcdefghijklmnopqrstuvwxyz ')}
}
INV_VOCAB: dict = {v: k for k, v in VOCAB.items()}
VOCAB_SIZE: int = len(VOCAB)   # 28
MASK_ID: int = VOCAB['[MASK]']  # 0
NEG_INF: float = -1e9


# ---------------------------------------------------------------------------
# Text8 Download and Tokenization
# ---------------------------------------------------------------------------

def _download_text8(cache_dir: str) -> dict:
    """Download and split text8. Cached to disk after first download.

    Returns dict with keys 'train', 'val', 'test' mapping to raw strings.
    Splits: train = first 90M chars, val = 90–95M, test = 95–100M.
    """
    raw_dir = os.path.join(cache_dir, 'raw')
    done_flag = os.path.join(cache_dir, 'done.flag')
    split_paths = {
        'train': os.path.join(raw_dir, 'train.txt'),
        'val':   os.path.join(raw_dir, 'val.txt'),
        'test':  os.path.join(raw_dir, 'test.txt'),
    }
    if not os.path.exists(done_flag):
        os.makedirs(raw_dir, exist_ok=True)
        zip_path = os.path.join(raw_dir, 'text8.zip')
        if not os.path.exists(zip_path):
            print('Downloading text8 from mattmahoney.net ...')
            urllib.request.urlretrieve('http://mattmahoney.net/dc/text8.zip', zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            raw = zf.read('text8').decode('utf-8')
        raw_splits = {
            'train': raw[:90_000_000],
            'val':   raw[90_000_000:95_000_000],
            'test':  raw[95_000_000:],
        }
        for name, path in split_paths.items():
            with open(path, 'w') as f:
                f.write(raw_splits[name])
        open(done_flag, 'w').close()
        print('Done.')
    return {name: open(path).read() for name, path in split_paths.items()}


def _tokenize(text: str) -> List[int]:
    """Convert a string to a list of token ids."""
    # Characters not in the vocab (digits, punctuation) are skipped
    return [VOCAB[ch] for ch in text.lower() if ch in VOCAB]


def decode(ids) -> str:
    """Convert a token-id sequence back to a human-readable string.

    Single-character tokens are rendered as themselves.
    [MASK] tokens are rendered as '_'.
    All other special tokens are rendered as a space.
    """
    out = []
    for i in ids:
        tok = INV_VOCAB.get(int(i), '?')
        out.append('_' if tok == '[MASK]' else tok)
    return ''.join(out)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Text8Dataset(Dataset):
    """Fixed-length non-overlapping chunks of text8 characters."""

    def __init__(self, text: str, seq_len: int):
        tokens = _tokenize(text)
        n = len(tokens) // seq_len
        tokens = tokens[:n * seq_len]
        self.data = torch.tensor(tokens, dtype=torch.long).view(n, seq_len)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]   # [L] int64


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def bits_per_char(loss_nats: float) -> float:
    """Convert average loss in nats to bits per character."""
    return loss_nats / math.log(2)


def word_error_rate(hyps: List[str], refs: List[str]) -> float:
    """Compute word-level WER over lists of hypothesis and reference strings."""
    total_words = total_errors = 0
    for hyp, ref in zip(hyps, refs):
        h_words = hyp.split()
        r_words = ref.split()
        total_words += len(r_words)
        total_errors += _word_edit_distance(h_words, r_words)
    return total_errors / max(total_words, 1)


def _word_edit_distance(a: List[str], b: List[str]) -> int:
    """Standard dynamic-programming word-level edit distance."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]
