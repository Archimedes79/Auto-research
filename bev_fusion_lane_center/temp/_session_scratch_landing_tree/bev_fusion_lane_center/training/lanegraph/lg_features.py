"""The input: the cached, frozen GaussianCaR BEV features of a frame (39 channels at 0.5 m, 200 x 200 cells: 32 decoder channels after PCA +
7 map logits), normalised by the training statistics and rounded to float16 exactly as the models were trained.

    store = FeatureStore()                 # the VAL / TEST feature files (`~/exp_gcar_b123/feats/val_*.npy`; one file holds both splits)
    rows = split_rows("val")               # the 873 VAL rows (data/splits.json)
    x = store(rows[:4])                    # (4, 39, 200, 200) float32
    store.tokens[row]                      # the nuScenes sample token of a row
"""
import json
from pathlib import Path

import numpy as np
import torch

from lg_config import PROJECT

FEATS = Path.home() / "exp_gcar_b123" / "feats"
STATS = PROJECT / "runs" / "refiner_stats_dec32_logit7.npz"


def split_rows(split: str) -> list:
    """the feature-file rows of a split ('val' or 'test')"""
    return [int(r) for r in json.load(open(PROJECT / "data" / "splits.json"))[split]]


class FeatureStore:
    def __init__(self, feats: Path = FEATS, stats: Path = STATS, part: str = "val"):
        self.mm = [np.load(feats / f"{part}_{g}.npy", mmap_mode="r") for g in ("dec32", "logit7")]
        z = np.load(stats)
        self.mean, self.std = z["mean"][:39, None, None].astype(np.float32), z["std"][:39, None, None].astype(np.float32)
        self.tokens = json.load(open(feats / f"{part}_tokens.json"))

    def __call__(self, rows: list) -> torch.Tensor:
        x = [np.concatenate([np.asarray(m[r], np.float32) for m in self.mm], 0) for r in rows]
        return torch.from_numpy(((np.stack(x) - self.mean) / self.std).astype(np.float16)).float()
