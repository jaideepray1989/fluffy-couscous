from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TransformerConfig:
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    include_time: bool = True
    deltas: bool = True
    num_classes: int = 10


class StrokeTransformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        input_dim = 3 if (cfg.deltas and not cfg.include_time) else 4
        self.input_proj = nn.Linear(input_dim, cfg.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.cls_head = nn.Linear(cfg.d_model, cfg.num_classes)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        # mean pool over valid positions
        if src_key_padding_mask is not None:
            mask = (~src_key_padding_mask).unsqueeze(-1)
            h = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            h = h.mean(dim=1)
        return self.cls_head(h)
