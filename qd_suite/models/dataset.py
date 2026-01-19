from __future__ import annotations

from typing import Dict, List, Sequence

import torch
from torch.utils.data import Dataset

from qd_suite.data.dataset import Sample
from qd_suite.repr.pointseq import PointSequence
from qd_suite.repr.tokenize import to_tokens


class PointSeqDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], class_to_idx: Dict[str, int], include_time: bool = True, deltas: bool = True):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.include_time = include_time
        self.deltas = deltas

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        tokens = to_tokens(sample.sequence, include_time=self.include_time, deltas=self.deltas)
        tensor = torch.tensor(tokens, dtype=torch.float32)
        label = self.class_to_idx[sample.label]
        return tensor, label


def collate_pad(batch, pad_value: float = 0.0):
    tensors, labels = zip(*batch)
    lengths = torch.tensor([t.shape[0] for t in tensors], dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=pad_value)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels


def pad_batch(seqs: Sequence[PointSequence], include_time: bool = True, deltas: bool = True, pad_value: float = 0.0):
    tensors = [torch.tensor(to_tokens(s, include_time=include_time, deltas=deltas), dtype=torch.float32) for s in seqs]
    lengths = torch.tensor([t.shape[0] for t in tensors], dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=pad_value)
    return padded, lengths
