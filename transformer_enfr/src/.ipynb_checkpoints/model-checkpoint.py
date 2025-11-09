
import math
import torch
import torch.nn as nn
from .utils import PositionalEncoding

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None):
        B, Tq, D = q.size()
        B, Tk, D = k.size()
        # Linear
        q = self.q_proj(q)  # (B, Tq, D)
        k = self.k_proj(k)  # (B, Tk, D)
        v = self.v_proj(v)  # (B, Tk, D)
        # Split heads
        q = q.view(B, Tq, self.nhead, self.d_k).transpose(1,2)  # (B, H, Tq, d_k)
        k = k.view(B, Tk, self.nhead, self.d_k).transpose(1,2)  # (B, H, Tk, d_k)
        v = v.view(B, Tk, self.nhead, self.d_k).transpose(1,2)  # (B, H, Tk, d_k)
        # Scaled dot-product
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B,H,Tq,Tk)

        if attn_mask is not None:
            # attn_mask: (Tq, Tk) or (B, Tq, Tk) or (1, Tq, Tk), values: 0 mask, 1 keep
            if attn_mask.dim() == 2:
                scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
            elif attn_mask.dim() == 3:
                scores = scores.masked_fill(attn_mask.unsqueeze(1) == 0, float('-inf'))
            else:
                scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        if key_padding_mask is not None:
            # key_padding_mask: (B, Tk) True for PAD
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,Tk)
            scores = scores.masked_fill(mask, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        ctx = torch.matmul(attn, v)  # (B,H,Tq,d_k)
        # Concatenate heads
        ctx = ctx.transpose(1,2).contiguous().view(B, Tq, self.d_model)  # (B,Tq,D)
        out = self.o_proj(ctx)
        return out

class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ff = FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        # Self-attention
        attn_out = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ff = FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mem, tgt_mask=None, tgt_key_padding_mask=None, mem_key_padding_mask=None):
        # Masked self-attention
        self_attn_out = self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        x = self.norm1(x + self.dropout(self_attn_out))
        # Cross attention
        cross_attn_out = self.cross_attn(x, mem, mem, key_padding_mask=mem_key_padding_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))
        # FFN
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x

class TransformerED(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=256, nhead=4, num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=512, dropout=0.1, max_len=512):
        super().__init__()
        self.src_embed = nn.Embedding(len(src_vocab), d_model)
        self.tgt_embed = nn.Embedding(len(tgt_vocab), d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_decoder_layers)
        ])

        self.generator = nn.Linear(d_model, len(tgt_vocab))

    def encode(self, src_ids, src_key_padding_mask):
        x = self.src_embed(src_ids)  # (B, S, D)
        x = self.pos_enc(x)
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x  # memory

    def decode(self, tgt_ids, memory, tgt_mask, tgt_key_padding_mask, mem_key_padding_mask):
        y = self.tgt_embed(tgt_ids)
        y = self.pos_enc(y)
        for layer in self.decoder_layers:
            y = layer(y, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                      mem_key_padding_mask=mem_key_padding_mask)
        return y

    def forward(self, src_ids, tgt_in_ids, src_key_padding_mask=None, tgt_key_padding_mask=None, tgt_mask=None):
        mem = self.encode(src_ids, src_key_padding_mask)
        dec = self.decode(tgt_in_ids, mem, tgt_mask, tgt_key_padding_mask, src_key_padding_mask)
        logits = self.generator(dec)
        return logits
