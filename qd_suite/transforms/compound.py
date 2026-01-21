from __future__ import annotations

from ..repr.pointseq import PointSequence
from . import dropout, noise, resample, affine, segment, reorder


def dropout_jitter(seq: PointSequence, p: float = 0.5, sigma: float = 0.05, seed: int | None = None) -> PointSequence:
    seq = dropout.point_dropout(seq, p=p, seed=seed)
    seq = dropout.stroke_dropout(seq, p=p, seed=seed)
    seq = noise.jitter_xy(seq, sigma=sigma, seed=seed)
    return seq


def resample_rotate(seq: PointSequence, N: int = 8, degrees: float = 30.0, seed: int | None = None) -> PointSequence:
    seq = resample.resample_uniform(seq, N=N)
    seq = affine.rotate(seq, degrees=degrees)
    return seq


def split_shuffle(seq: PointSequence, k: int = 3, window: int = 5, seed: int | None = None) -> PointSequence:
    seq = segment.split_stroke(seq, k=k, seed=seed)
    seq = reorder.local_shuffle_strokes(seq, window=window, seed=seed)
    return seq


def quant_scale(seq: PointSequence, bits: int = 3, factor: float = 0.8) -> PointSequence:
    seq = noise.quantize_xy(seq, bits=bits)
    seq = affine.scale(seq, factor=factor)
    return seq
