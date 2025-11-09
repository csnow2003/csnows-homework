import math
import torch
import torch.nn as nn
from .utils import PositionalEncoding

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, use_multihead=True):
        super().__init__()
        assert d_model % nhead == 0 or not use_multihead, "d_model must be divisible by nhead when use_multihead=True"
        self.d_model = d_model
        self.nhead = nhead if use_multihead else 1
        self.d_k = d_model // self.nhead if use_multihead else d_model
        self.use_multihead = use_multihead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None):
        batch_size = q.size(0)
        seq_len = q.size(1)

        # 投影到Q, K, V
        q = self.q_proj(q)  # (B, S, D)
        k = self.k_proj(k)
        v = self.v_proj(v)

        if self.use_multihead:
            # 多头分割
            q = q.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)  # (B, H, S, Dk)
            k = k.view(batch_size, k.size(1), self.nhead, self.d_k).transpose(1, 2)
            v = v.view(batch_size, v.size(1), self.nhead, self.d_k).transpose(1, 2)
        else:
            # 单头注意力
            q = q.unsqueeze(1)  # (B, 1, S, D)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)

        # 缩放点积注意力
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, H, S, S)

        # 应用掩码
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask.unsqueeze(1) == 0, -1e9)
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2) == 1, -1e9)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 注意力加权和
        output = torch.matmul(attn_weights, v)  # (B, H, S, Dk)

        if self.use_multihead:
            # 多头合并
            output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        else:
            output = output.squeeze(1)  # (B, S, D)

        # 输出投影
        output = self.o_proj(output)
        return output

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
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, 
                 use_multihead=True, use_residual=True, use_layernorm=True):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout, use_multihead)
        self.ff = FeedForward(d_model, dim_feedforward, dropout)
        
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm
        
        if use_layernorm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        # Self-attention
        attn_out = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        
        # 残差连接和LayerNorm
        if self.use_residual and self.use_layernorm:
            x = self.norm1(x + self.dropout(attn_out))
        elif self.use_residual:
            x = x + self.dropout(attn_out)
        elif self.use_layernorm:
            x = self.norm1(self.dropout(attn_out))
        else:
            x = self.dropout(attn_out)
        
        # FFN
        ff_out = self.ff(x)
        
        # 残差连接和LayerNorm
        if self.use_residual and self.use_layernorm:
            x = self.norm2(x + self.dropout(ff_out))
        elif self.use_residual:
            x = x + self.dropout(ff_out)
        elif self.use_layernorm:
            x = self.norm2(self.dropout(ff_out))
        else:
            x = self.dropout(ff_out)
            
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, 
                 use_multihead=True, use_residual=True, use_layernorm=True):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout, use_multihead)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout, use_multihead)
        self.ff = FeedForward(d_model, dim_feedforward, dropout)
        
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm
        
        if use_layernorm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mem, tgt_mask=None, tgt_key_padding_mask=None, mem_key_padding_mask=None):
        # Masked self-attention
        self_attn_out = self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        
        # 残差连接和LayerNorm
        if self.use_residual and self.use_layernorm:
            x = self.norm1(x + self.dropout(self_attn_out))
        elif self.use_residual:
            x = x + self.dropout(self_attn_out)
        elif self.use_layernorm:
            x = self.norm1(self.dropout(self_attn_out))
        else:
            x = self.dropout(self_attn_out)
        
        # Cross attention
        cross_attn_out = self.cross_attn(x, mem, mem, key_padding_mask=mem_key_padding_mask)
        
        # 残差连接和LayerNorm
        if self.use_residual and self.use_layernorm:
            x = self.norm2(x + self.dropout(cross_attn_out))
        elif self.use_residual:
            x = x + self.dropout(cross_attn_out)
        elif self.use_layernorm:
            x = self.norm2(self.dropout(cross_attn_out))
        else:
            x = self.dropout(cross_attn_out)
        
        # FFN
        ff_out = self.ff(x)
        
        # 残差连接和LayerNorm
        if self.use_residual and self.use_layernorm:
            x = self.norm3(x + self.dropout(ff_out))
        elif self.use_residual:
            x = x + self.dropout(ff_out)
        elif self.use_layernorm:
            x = self.norm3(self.dropout(ff_out))
        else:
            x = self.dropout(ff_out)
            
        return x

class TransformerED(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=256, nhead=4, 
                 num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=512, dropout=0.1, max_len=512,
                 use_multihead=True, use_positional_encoding=True,
                 use_residual=True, use_layernorm=True,
                 use_decoder=True):
        super().__init__()
        self.src_embed = nn.Embedding(len(src_vocab), d_model)
        self.tgt_embed = nn.Embedding(len(tgt_vocab), d_model)
        
        self.use_positional_encoding = use_positional_encoding
        self.use_decoder = use_decoder
        
        if use_positional_encoding:
            self.pos_enc = PositionalEncoding(d_model, max_len=max_len)

        # 编码器层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, dim_feedforward, dropout,
                        use_multihead, use_residual, use_layernorm) 
            for _ in range(num_encoder_layers)
        ])
        
        # 解码器层（可选）
        if use_decoder:
            self.decoder_layers = nn.ModuleList([
                DecoderLayer(d_model, nhead, dim_feedforward, dropout,
                            use_multihead, use_residual, use_layernorm) 
                for _ in range(num_decoder_layers)
            ])
        
        # 输出层
        self.generator = nn.Linear(d_model, len(tgt_vocab))

    def encode(self, src_ids, src_key_padding_mask):
        x = self.src_embed(src_ids)  # (B, S, D)
        if self.use_positional_encoding:
            x = self.pos_enc(x)
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x  # memory

    def decode(self, tgt_ids, memory, tgt_mask, tgt_key_padding_mask, mem_key_padding_mask):
        if not self.use_decoder:
            # 如果没有解码器，直接使用编码器输出
            return memory
        
        y = self.tgt_embed(tgt_ids)
        if self.use_positional_encoding:
            y = self.pos_enc(y)
        for layer in self.decoder_layers:
            y = layer(y, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                      mem_key_padding_mask=mem_key_padding_mask)
        return y

    def forward(self, src_ids, tgt_in_ids, src_key_padding_mask=None, tgt_key_padding_mask=None, tgt_mask=None):
        mem = self.encode(src_ids, src_key_padding_mask)
        
        if self.use_decoder:
            dec = self.decode(tgt_in_ids, mem, tgt_mask, tgt_key_padding_mask, src_key_padding_mask)
        else:
            # 无解码器时，重复编码器输出用于序列生成
            dec = mem[:, :tgt_in_ids.size(1), :]
        
        logits = self.generator(dec)
        return logits