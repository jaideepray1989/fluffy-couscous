from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for batch-first tensors."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model)
        x = x + self.pe[: x.size(1)].transpose(0, 1)
        return self.dropout(x)


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.score = nn.Linear(d_model, 1, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.score(h).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        attn = torch.softmax(scores, dim=1)
        attn = self.dropout(attn)
        return (h * attn.unsqueeze(-1)).sum(dim=1)


@dataclass
class TransformerConfig:
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    input_dropout: float = 0.0
    pooling: str = "mean"
    include_time: bool = True
    deltas: bool = True
    num_classes: int = 10


class StrokeTransformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        input_dim = 4 if cfg.include_time else 3
        self.input_proj = nn.Linear(input_dim, cfg.d_model)
        self.input_norm = nn.LayerNorm(cfg.d_model)
        self.input_dropout = nn.Dropout(p=cfg.input_dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers, norm=nn.LayerNorm(cfg.d_model))
        self.pos_encoder = PositionalEncoding(cfg.d_model, dropout=cfg.dropout)
        self.pooling = cfg.pooling
        if cfg.pooling == "attn":
            self.attn_pool = AttentionPooling(cfg.d_model, dropout=cfg.dropout)
        elif cfg.pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        self.cls_head = nn.Linear(cfg.d_model, cfg.num_classes)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.input_norm(h)
        h = self.input_dropout(h)
        if self.pooling == "cls":
            cls = self.cls_token.expand(h.size(0), -1, -1)
            h = torch.cat([cls, h], dim=1)
            if src_key_padding_mask is not None:
                cls_mask = torch.zeros((h.size(0), 1), dtype=src_key_padding_mask.dtype, device=src_key_padding_mask.device)
                src_key_padding_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)
        h = self.pos_encoder(h)
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        if self.pooling == "cls":
            h = h[:, 0]
        elif self.pooling == "attn":
            h = self.attn_pool(h, src_key_padding_mask)
        else:
            if src_key_padding_mask is not None:
                mask = (~src_key_padding_mask).unsqueeze(-1)
                h = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            else:
                h = h.mean(dim=1)
        return self.cls_head(h)
