
import math
import torch
import random
import numpy as np

SPECIAL_TOKENS = {"pad": "<pad>", "bos": "<bos>", "eos": "<eos>", "unk": "<unk>"}

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:, :T, :]

def subsequent_mask(sz: int) -> torch.Tensor:
    """Mask out subsequent positions. 1: keep, 0: mask"""
    attn_shape = (1, sz, sz)
    subsequent = torch.triu(torch.ones(attn_shape), diagonal=1).bool()
    return (~subsequent).float()  # (1, T, T)

def lengths_to_mask(lengths, max_len=None):
    if max_len is None:
        max_len = max(lengths)
    idxs = torch.arange(0, max_len, device=lengths.device).unsqueeze(0)
    mask = idxs < lengths.unsqueeze(1)  # (B, T)
    return mask
