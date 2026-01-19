from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from qd_suite.repr.pointseq import PointSequence
from qd_suite.repr.tokenize import to_tokens


def seq_to_tensor(seq: PointSequence, include_time: bool = True, deltas: bool = True) -> torch.Tensor:
    tokens = to_tokens(seq, include_time=include_time, deltas=deltas)
    return torch.tensor(tokens, dtype=torch.float32)


def pad_batch(seqs: List[PointSequence], include_time: bool = True, deltas: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    tensors = [seq_to_tensor(s, include_time=include_time, deltas=deltas) for s in seqs]
    lengths = torch.tensor([t.shape[0] for t in tensors], dtype=torch.long)
    padded = nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    return padded, lengths


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        h_last = h[-1]
        return self.fc(h_last)


@dataclass
class GRUConfig:
    input_dim: int = 3  # dx, dy, pen (and optionally dt if include_time)
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    include_time: bool = True
    deltas: bool = True


def build_model(num_classes: int, config: GRUConfig | None = None) -> GRUClassifier:
    cfg = config or GRUConfig()
    input_dim = 3 if (cfg.deltas and not cfg.include_time) else 4
    model = GRUClassifier(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_classes=num_classes,
        dropout=cfg.dropout,
    )
    return model
