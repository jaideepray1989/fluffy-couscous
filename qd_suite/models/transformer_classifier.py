from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from qd_suite.models.gru_classifier import GRUConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim: int, model_dim: int, nhead: int, num_layers: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.model_dim = model_dim
        self.embedding = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(model_dim, nhead, model_dim * 4, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(model_dim, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Create a mask for the padded elements
        mask = torch.arange(x.size(1), device=x.device)[None, :] >= lengths[:, None]
        
        embedded = self.embedding(x) * math.sqrt(self.model_dim)
        pos_encoded = self.pos_encoder(embedded)
        
        # Apply the transformer encoder
        encoder_output = self.transformer_encoder(pos_encoded, src_key_padding_mask=mask)
        
        # Pool the output (masked mean)
        valid = (~mask).unsqueeze(-1)
        pooled = (encoder_output * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        
        return self.fc(pooled)


@dataclass
class TransformerConfig(GRUConfig):
    model_dim: int = 128
    nhead: int = 4


def build_model(num_classes: int, config: TransformerConfig | None = None) -> TransformerClassifier:
    cfg = config or TransformerConfig()
    input_dim = 4 if cfg.include_time else 3
    model = TransformerClassifier(
        input_dim=input_dim,
        model_dim=cfg.model_dim,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        num_classes=num_classes,
        dropout=cfg.dropout,
    )
    return model
