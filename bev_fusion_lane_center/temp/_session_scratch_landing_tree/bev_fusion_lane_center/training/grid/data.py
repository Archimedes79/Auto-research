"""Data of the grid head: the 39 frozen BEV channels of a frame and its ground-truth lane points (light: no point-graph / Dijkstra of `BuilderData.__getitem__`).

GridData(split) with split "train" (8414 frames) or "val_clean" (873 frames, the VAL split through `BuilderData.rows`: the stored feature files also hold the TEST rows and are
never iterated). Item: x (39, 200, 200) float16 (normalised as `BuilderData`: the 39 channels dec32 + logit7, the old dense head's 3 channels are not loaded), g05 / t05 (G, 2) the
GT lane points every 0.5 m inside |x|, |y| < 48 and their travel tangents, row (the dataset row), k (position in the split), scene.
`collate` pads g05 with 1e4 and t05 with 0 and gives the mask m05. Paths, rows, GT graphs and the normalisation come from `BuilderData` (one source of truth).
"""
import os
import sys

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "training", "data"))
import p16_data as PD  # noqa: E402

BOX = 48.0


class GridData(torch.utils.data.Dataset):
    def __init__(self, split="train", ks=None):
        self.bd = PD.BuilderData(split, ks=ks, train=(split == "train"))
        self.mm = None

    def __len__(self):
        return len(self.bd.ks)

    def _open(self):
        if self.mm is None:
            self.mm = [np.load(f"{PD.D}/feats/{self.bd.part}_{g}.npy", mmap_mode="r") for g in ("dec32", "logit7")]

    def __getitem__(self, i):
        import graph_eval as GE
        self._open()
        k = int(self.bd.ks[i])
        row = int(self.bd.rows[k])
        x = np.concatenate([np.asarray(m[row], np.float32) for m in self.mm], 0)
        x = ((x - self.bd.mean[:39, None, None]) / self.bd.std[:39, None, None]).astype(np.float16)
        P, T, _ = GE.dense_points(self.bd.gt[row], 0.5)
        P, T = np.asarray(P, np.float32).reshape(-1, 2), np.asarray(T, np.float32).reshape(-1, 2)
        keep = (np.abs(P[:, 0]) < BOX) & (np.abs(P[:, 1]) < BOX)
        return dict(x=torch.from_numpy(x), g05=torch.from_numpy(P[keep]), t05=torch.from_numpy(T[keep]), k=k, row=row)


def collate(b):
    G = max(1, max(len(o["g05"]) for o in b))
    B = len(b)
    g05, t05, m05 = torch.full((B, G, 2), 1e4), torch.zeros(B, G, 2), torch.zeros(B, G)
    for i, o in enumerate(b):
        n = len(o["g05"])
        g05[i, :n], t05[i, :n], m05[i, :n] = o["g05"], o["t05"], 1.0
    return dict(x=torch.stack([o["x"] for o in b]), g05=g05, t05=t05, m05=m05, k=torch.tensor([o["k"] for o in b]), row=torch.tensor([o["row"] for o in b]))


def loader(split="train", bs=8, workers=3, shuffle=True, ks=None):
    ds = GridData(split, ks)
    return ds, torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers, collate_fn=collate, drop_last=shuffle,
                                           persistent_workers=workers > 0, prefetch_factor=2 if workers > 0 else None)
