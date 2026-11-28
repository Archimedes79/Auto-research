"""Data of the E271 grid head (the pre-training of stage 0, `training/train/qgraph_train.py` phase 1): the 39 frozen BEV channels of a frame and its
ground-truth lane points, from the graph ground truth of the frames (`frames.Frames(...).gt`, the stored graph GT of B123), not from the lane GT of GtGenerator.

    ds = GridData("train" | "val_clean", ks=None);  dl = loader(ds, bs=8, workers=3, batches=None)

Item: x (39, 200, 200) float16 (normalised with `frames.norm_stats`), g05 / t05 (G, 2) the GT lane points every 0.5 m inside |x|, |y| < 48 m and their travel
tangents, row (the row of the part), k (the position in the split). `collate` pads g05 with 1e4 and t05 with 0 and gives the mask m05: the batch of
`losses.grid_loss`. The rows of a split come through `frames.Frames` (the stored feature files also hold the TEST rows, which are never iterated).
Test: net/tests/test_qgraph_train.py"""
import numpy as np
import torch

from lane_center.net.prep import frames as FR
from lane_center.net.prep.polylines import dense_points

BOX = 48.0


class GridData(torch.utils.data.Dataset):
    def __init__(self, split="train", ks=None):
        self.bd = FR.Frames(split, ks=ks)
        self.mm = None

    def __len__(self):
        return len(self.bd.ks)

    def _open(self):
        if self.mm is None:
            self.mm = [np.load(f"{FR.D}/feats/{self.bd.part}_{g}.npy", mmap_mode="r") for g in ("dec32", "logit7")]

    def __getitem__(self, i):
        self._open()
        k = int(self.bd.ks[i])
        row = int(self.bd.rows[k])
        x = np.concatenate([np.asarray(m[row], np.float32) for m in self.mm], 0)
        x = ((x - self.bd.mean[:39, None, None]) / self.bd.std[:39, None, None]).astype(np.float16)
        P, T, _ = dense_points(self.bd.gt[row], 0.5)
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


def loader(ds, bs=8, workers=3, batches=None):
    """the batches of `ds` in order (`batches`: a batch sampler, e.g. the trainer's step order, else the frames in order, batch `bs`)"""
    kw = dict(batch_sampler=batches) if batches is not None else dict(batch_size=bs, shuffle=False)
    return torch.utils.data.DataLoader(ds, num_workers=workers, collate_fn=collate, persistent_workers=workers > 0, prefetch_factor=2 if workers > 0 else None,
                                       **kw)
