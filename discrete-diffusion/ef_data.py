"""
Data utilities for the EditFlow workshop notebook.

Vocabulary (31 tokens):
  [MASK] = 0    a–z = 1–26    space = 27
  [BOS]  = 28   [EOS] = 29    [PAD]  = 30

Each sequence is wrapped with [BOS] and [EOS].  The dataset returns fixed-length
tensors of shape (content_len + 2,) where positions 0 and -1 are BOS/EOS.
"""

import os, math, urllib.request, zipfile
import torch
from torch.utils.data import Dataset

# ── Vocabulary ───────────────────────────────────────────────────────────────

VOCAB = {
    '[MASK]': 0,
    **{chr(ord('a') + i): i + 1 for i in range(26)},
    ' ':      27,
    '[BOS]':  28,
    '[EOS]':  29,
    '[PAD]':  30,
}
ID2TOK = {v: k for k, v in VOCAB.items()}

VOCAB_SIZE = 31
MASK_ID    = VOCAB['[MASK]']
BOS_ID     = VOCAB['[BOS]']
EOS_ID     = VOCAB['[EOS]']
PAD_ID     = VOCAB['[PAD]']
NEG_INF    = -1e9

# ── Tokenizer ────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list:
    return [VOCAB[ch] for ch in text.lower() if ch in VOCAB]


def decode(token_ids) -> str:
    """Decode token ids → string.  MASK → '_', BOS/EOS/PAD → omitted."""
    if hasattr(token_ids, 'tolist'):
        token_ids = token_ids.tolist()
    out = []
    for tid in token_ids:
        if tid == MASK_ID:
            out.append('_')
        elif tid in (BOS_ID, EOS_ID, PAD_ID):
            continue
        else:
            out.append(ID2TOK.get(tid, '?'))
    return ''.join(out)

# ── text8 download ────────────────────────────────────────────────────────────

TEXT8_URL = 'http://mattmahoney.net/dc/text8.zip'

def _download_text8(cache_dir: str) -> dict:
    """Download and split text8. Returns {'train': str, 'val': str, 'test': str}."""
    os.makedirs(cache_dir, exist_ok=True)
    txt_path = os.path.join(cache_dir, 'text8.txt')
    if not os.path.exists(txt_path):
        zip_path = os.path.join(cache_dir, 'text8.zip')
        if not os.path.exists(zip_path):
            print('Downloading text8 ...')
            urllib.request.urlretrieve(TEXT8_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            with open(txt_path, 'w') as f:
                f.write(zf.read('text8').decode('utf-8'))
        print('Done.')
    text = open(txt_path).read()
    return {
        'train': text[:90_000_000],
        'val':   text[90_000_000:95_000_000],
        'test':  text[95_000_000:],
    }

# ── Dataset ───────────────────────────────────────────────────────────────────

class Text8Dataset(Dataset):
    """
    Chunk text8 into fixed-length sequences wrapped with [BOS] and [EOS].

    Each item is a tensor of length (content_len + 2):
        [BOS] c_0 c_1 ... c_{L-1} [EOS]
    """

    def __init__(self, text: str, content_len: int):
        tokens = _tokenize(text)
        self.seq_len = content_len + 2
        self.chunks  = []
        for i in range(0, len(tokens) - content_len + 1, content_len):
            chunk = tokens[i:i + content_len]
            if len(chunk) == content_len:
                self.chunks.append([BOS_ID] + chunk + [EOS_ID])

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return torch.tensor(self.chunks[idx], dtype=torch.long)

# ── Metrics ───────────────────────────────────────────────────────────────────

def bits_per_char(loss_nats: float) -> float:
    return loss_nats / math.log(2)
