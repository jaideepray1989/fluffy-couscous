from __future__ import annotations

import os
from typing import List

import torch
from sklearn.neighbors import KNeighborsClassifier

from qd_suite.adapters.base import ModelAdapter
from qd_suite.data.dataset import load_from_root
from qd_suite.models.contrastive import ContrastiveConfig, ContrastiveModel, pad_batch
from qd_suite.repr.pointseq import PointSequence


class ContrastiveKNNAdapter(ModelAdapter):
    def __init__(self, checkpoint_path: str, support_data: str, support_classes: List[str], support_limit: int = 500):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        cfg = ContrastiveConfig(**ckpt["config"])
        self.model = ContrastiveModel(cfg)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.include_time = cfg.include_time
        self.deltas = cfg.deltas
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.class_to_idx = ckpt["class_to_idx"]
        # build support embeddings
        ds = load_from_root(support_data, support_classes, limit_per_class=support_limit)
        seqs = [s.sequence for s in ds.samples]
        labels = [s.label for s in ds.samples]
        x, lengths = pad_batch(seqs, include_time=self.include_time, deltas=self.deltas)
        x, lengths = x.to(self.device), lengths.to(self.device)
        with torch.no_grad():
            emb = self.model.embed(x, lengths).cpu()
        self.knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
        self.knn.fit(emb, labels)

    def predict(self, batch: List[PointSequence]) -> List[str]:
        x, lengths = pad_batch(batch, include_time=self.include_time, deltas=self.deltas)
        x, lengths = x.to(self.device), lengths.to(self.device)
        with torch.no_grad():
            emb = self.model.embed(x, lengths).cpu()
        return self.knn.predict(emb).tolist()


def get_adapter():
    ckpt = os.environ.get("CONTRASTIVE_CHECKPOINT", "runs/contrastive/contrastive.pt")
    support_data = os.environ.get("CONTRASTIVE_SUPPORT_DATA", "data/raw")
    support_classes = os.environ.get("CONTRASTIVE_SUPPORT_CLASSES")
    if not support_classes:
        raise RuntimeError("Set CONTRASTIVE_SUPPORT_CLASSES env var (comma-separated classes).")
    support_limit = int(os.environ.get("CONTRASTIVE_SUPPORT_LIMIT", "500"))
    classes = [c.strip() for c in support_classes.split(",") if c.strip()]
    return ContrastiveKNNAdapter(ckpt, support_data, classes, support_limit=support_limit)
