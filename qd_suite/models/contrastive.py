from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from qd_suite.repr.tokenize import to_tokens
from qd_suite.repr.pointseq import PointSequence


@dataclass
class ContrastiveConfig:
    hidden_dim: int = 128
    proj_dim: int = 64
    num_layers: int = 2
    include_time: bool = True
    deltas: bool = True
    margin: float = 0.2


def seq_to_tensor(seq: PointSequence, include_time: bool = True, deltas: bool = True) -> torch.Tensor:
    tokens = to_tokens(seq, include_time=include_time, deltas=deltas)
    return torch.tensor(tokens, dtype=torch.float32)


def pad_batch(seqs: Sequence[PointSequence], include_time: bool = True, deltas: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    tensors = [seq_to_tensor(s, include_time=include_time, deltas=deltas) for s in seqs]
    lengths = torch.tensor([t.shape[0] for t in tensors], dtype=torch.long)
    padded = nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    return padded, lengths


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return h[-1]


class ContrastiveModel(nn.Module):
    def __init__(self, cfg: ContrastiveConfig):
        super().__init__()
        input_dim = 3 if (cfg.deltas and not cfg.include_time) else 4
        self.encoder = Encoder(input_dim=input_dim, hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers)
        self.proj = nn.Linear(cfg.hidden_dim, cfg.proj_dim)
        self.cfg = cfg

    def embed(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x, lengths)
        z = self.proj(h)
        return F.normalize(z, dim=-1)

    def loss(self, z_anchor: torch.Tensor, z_pos: torch.Tensor, z_neg: torch.Tensor, margin: float | None = None) -> torch.Tensor:
        m = margin if margin is not None else self.cfg.margin
        pos_dist = (z_anchor - z_pos).pow(2).sum(dim=-1)
        neg_dist = (z_anchor - z_neg).pow(2).sum(dim=-1)
        return F.relu(pos_dist - neg_dist + m).mean()
