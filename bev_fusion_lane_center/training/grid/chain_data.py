"""Data of the grid chain (E272): a short sequence of frames of one scene with the ego motion between them, the SD map of the LAST frame and the road labels of its GT lane points.

ChainData(split, T=3): item i = the frame k = ks[i] and its T - 1 predecessors through `poses["prev"]` (2 Hz keyframes of the same scene). A missing predecessor repeats the earliest
frame that exists (step 0, flag `has[t]` = 0: the hold sees the same frame twice, a no-op). Item: x (T, 39, 200, 200) float16 (normalised as `data.GridData`), g05 / t05 per frame
(the GT lane points of every frame, for the per-frame head loss), `step` (T - 1, 3) = pose of frame t + 1 in frame t as (tx, ty, yaw) in the axes of the raster (rows = x right, columns =
y forward, the frame of the graphs), the SD map tokens of the last frame (`assign.map_tokens`, grid frame x forward, y left) and `gmask` (G, R + 1) bool: the roads that are a PLAUSIBLE
label of every GT point of the last frame (the road token nearest to the point, or any road within 0.5 m of it, with a heading within 30 degrees; no such road within 3 m: the dustbin
bit R). The SD map holds duplicate roads (two carriageways within 0.5 m), so a label is a set, not one id.
Poses: R, t map the ego frame to the world (as `p16_builder.warp_prior`), so p_prev = R_p^T (R_c p_cur + t_c - t_p).
"""
import os
import sys

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("training/grid", "training/data"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import assign as AS  # noqa: E402
import data as GD  # noqa: E402
import p16_data as PD  # noqa: E402


def ego_step(pose, row, prow):
    """pose of the frame `row` in the frame `prow` (the previous one) as (tx, ty, yaw): p_prev = R(yaw) p_cur + t"""
    Rc, tc = pose["R"][row].astype(np.float64), pose["t"][row].astype(np.float64)
    Rp, tp = pose["R"][prow].astype(np.float64), pose["t"][prow].astype(np.float64)
    Rr = Rp.T @ Rc
    t = Rp.T @ (tc - tp)
    return np.array([t[0], t[1], np.arctan2(Rr[1, 0], Rr[0, 0])], np.float32)


def road_labels(g05, t05, mp, dmax=3.0, slack=0.5, ang_deg=30.0):
    """GT points (G, 2) and tangents (G, 2) in the metric frame (x right, y forward) + the map tokens `mp` of `assign.map_tokens` -> bool (G, R + 1): the plausible roads of every point,
    the dustbin bit R when no road with a matching heading lies within dmax"""
    G, R = len(g05), int(mp["n_ids"])
    out = np.zeros((G, R + 1), bool)
    if G == 0 or len(mp["pos"]) == 0:
        out[:, R] = True
        return out
    pos = np.stack([g05[:, 1], -g05[:, 0]], 1)                                  # metric frame -> grid frame (x forward, y left)
    hdg = np.stack([t05[:, 1], -t05[:, 0]], 1)
    d = np.linalg.norm(pos[:, None, :] - mp["pos"][None, :, :], axis=-1)       # (G, M)
    okh = hdg @ mp["dir"].T > np.cos(np.deg2rad(ang_deg))
    d = np.where(okh, d, np.inf)
    best = np.full((G, R), np.inf)
    for r in np.unique(mp["rid"]):
        best[:, r] = d[:, mp["rid"] == r].min(1)
    dm = best.min(1)
    has = dm <= dmax
    out[:, :R] = (best <= dm[:, None] + slack) & has[:, None]
    out[:, R] = ~has
    return out


class ChainData(torch.utils.data.Dataset):
    """item i = the chain that ENDS at the frame at position i of the split (all frames of the split are positions: `GridData` with ks = None)"""

    def __init__(self, split="train", T=3):
        self.T = T
        self.gd = GD.GridData(split)
        self.bd = self.gd.bd
        self.pose = dict(np.load(f"{PD.D}/poses_{self.bd.part}.npz"))
        self.row2k = {int(r): k for k, r in enumerate(self.bd.rows)}
        import p8_lib as L
        self.sds = L.cache("sd", self.bd.part)

    def __len__(self):
        return len(self.gd)

    def chain(self, k):
        """positions k_0 .. k_{T-1} = k of the frames of the chain (the earliest existing one repeated where a predecessor is missing) and the has-flags"""
        ks, has = [k], []
        for _ in range(self.T - 1):
            row = int(self.bd.rows[ks[0]])
            prow = int(self.pose["prev"][row])
            pk = self.row2k.get(prow, -1) if prow >= 0 else -1
            has.insert(0, pk >= 0)
            ks.insert(0, pk if pk >= 0 else ks[0])
        return ks, has

    def __getitem__(self, k):
        k = int(k)
        ks, has = self.chain(k)
        items = [self.gd[kk] for kk in ks]
        steps = []
        for t in range(self.T - 1):
            ra, rb = int(self.bd.rows[ks[t + 1]]), int(self.bd.rows[ks[t]])
            steps.append(ego_step(self.pose, ra, rb) if has[t] else np.zeros(3, np.float32))
        row = items[-1]["row"]
        sd = self.sds[row]
        mp, gmask = None, None
        if sd is not None:
            mp = AS.map_tokens(dict(roads=sd["roads"], route=[]))
            gmask = road_labels(items[-1]["g05"].numpy().astype(np.float64), items[-1]["t05"].numpy().astype(np.float64), mp)
        return dict(x=torch.stack([o["x"] for o in items]), g05=[o["g05"] for o in items], t05=[o["t05"] for o in items], step=torch.from_numpy(np.stack(steps)),
                    has=torch.tensor(has, dtype=torch.float32), mp=mp, gmask=None if gmask is None else torch.from_numpy(gmask), k=k, row=row)


class _Sub(torch.utils.data.Dataset):
    def __init__(self, ds, ks):
        self.ds, self.ks = ds, np.asarray(ks)

    def __len__(self):
        return len(self.ks)

    def __getitem__(self, i):
        return self.ds[int(self.ks[i])]


def collate(b):
    T = b[0]["x"].shape[0]
    B = len(b)
    out = dict(x=torch.stack([o["x"] for o in b]), step=torch.stack([o["step"] for o in b]), has=torch.stack([o["has"] for o in b]),
               k=torch.tensor([o["k"] for o in b]), row=torch.tensor([o["row"] for o in b]), mp=[o["mp"] for o in b], gmask=[o["gmask"] for o in b])
    G = max(1, max(len(g) for o in b for g in o["g05"]))
    g05, t05, m05 = torch.full((B, T, G, 2), 1e4), torch.zeros(B, T, G, 2), torch.zeros(B, T, G)
    for i, o in enumerate(b):
        for t in range(T):
            n = len(o["g05"][t])
            g05[i, t, :n], t05[i, t, :n], m05[i, t, :n] = o["g05"][t], o["t05"][t], 1.0
    out.update(g05=g05, t05=t05, m05=m05)
    return out


def loader(split="train", T=3, bs=4, workers=3, shuffle=True, ks=None):
    """`ks` selects the positions of the LAST frames of the chains (all frames when None); the predecessors come from the whole split"""
    ds = ChainData(split, T=T)
    sub = _Sub(ds, np.arange(len(ds)) if ks is None else ks)
    return ds, torch.utils.data.DataLoader(sub, batch_size=bs, shuffle=shuffle, num_workers=workers, collate_fn=collate, drop_last=shuffle,
                                           persistent_workers=workers > 0, prefetch_factor=2 if workers > 0 else None)
