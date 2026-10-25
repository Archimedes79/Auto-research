"""Loop 16 data for the differentiable lane-graph builders (designs D1 - D8).
One item = one frame: the 42 cached BEV channels (39 frozen features + dense-head heat + direction field; out-of-fold heat / direction for training rows, as in E94 / E120),
the exact distance and direction fields of the ground truth (verification targets), and the ground-truth graph as tensors:
  g05 [Gm,2] dense lane points every 0.5 m (|x|,|y| < 48), t05 [Gm,2] travel tangent, n05 [Gm] index of the nearest 1 m point;
  g1 [Hm,2] lane points every 1 m with the directed point graph of graph_eval, dg [Hm,Hm] uint16 directed shortest-path distance in 0.1 m (65535 = unreachable, cap 40 m).
Usage: imported by p16_train.py / p16_gate.py / p16_diag.py."""
import bflc_paths as BP  # project paths from the location of this checkout
import os
import pickle
import sys

import numpy as np
import torch
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_eval as GE  # noqa: E402
import p10_infer as PI  # noqa: E402
import p8_lib as L  # noqa: E402

HOME = os.path.expanduser("~")
D = f"{HOME}/exp_gcar_b123"
# folder of <part>_gt_dist_exact.npy (the seed / direction / builder target); default = the stored b123 map (made with TF32 matrix products, D10).
# LCR_DIST_DIR=$HOME/exp_gcar_b123x/feats selects the CPU-exact rebuild (E251); the choice is part of the run command, so the ledger records it.
DIST_DIR = os.environ.get("LCR_DIST_DIR") or f"{HOME}/exp_crn_b123/feats"
OUT = BP.RUNS
UNREACH = 65535
DG_CAP = 40.0


def gt_tensors(g):
    """ground-truth graph (graph_eval format) -> numpy arrays of the item"""
    P, Dd, _ = GE.dense_points(g, 0.5)
    xy, edges, _, _ = GE.point_graph(g, 1.0)
    sp_, mg_ = GE.junctions(g)  # ground-truth split and merge landmarks (end of a piece with >= 2 successors / start of a piece with >= 2 predecessors)
    jz = dict(js=np.asarray(sp_, np.float32).reshape(-1, 2), jm=np.asarray(mg_, np.float32).reshape(-1, 2))
    if len(xy) == 0 or len(P) == 0:
        return dict(g05=np.zeros((0, 2), np.float32), t05=np.zeros((0, 2), np.float32), n05=np.zeros(0, np.int64), g1=np.zeros((0, 2), np.float32), dg=np.zeros((0, 0), np.uint16), **jz)
    n = len(xy)
    A = GE._csr(xy, edges)
    M = dijkstra(A, directed=True, limit=DG_CAP)
    dg = np.where(np.isfinite(M), np.round(np.nan_to_num(M, posinf=0.0) * 10.0), UNREACH).astype(np.uint16)
    n05 = cKDTree(xy).query(P)[1].astype(np.int64)
    return dict(g05=P.astype(np.float32), t05=Dd.astype(np.float32), n05=n05, g1=xy.astype(np.float32), dg=dg.reshape(n, n), **jz)


class BuilderData(torch.utils.data.Dataset):
    """split: 'train' (8414 frames, out-of-fold dense heads) | 'val_clean' (873) | 'test_clean' (833)"""

    def __init__(self, split="train", ks=None, train=True, with_prev=False):
        self.split, self.train, self.with_prev = split, train, with_prev
        z = np.load(f"{OUT}/refiner_stats_dec32_logit7.npz")
        self.mean, self.std = z["mean"].astype(np.float32), z["std"].astype(np.float32)
        self.mm = None
        if split == "train":
            self.part = "train"
            self.gt = L.cache("gt", "train")
            self.rows = np.arange(len(self.gt))
            self.ids = self.rows
        else:
            self.es = PI.EvalSet(split)
            self.part = self.es.part
            self.gt = self.es.gt
            self.rows = np.asarray(self.es.rows)
        self.ks = np.arange(len(self.rows)) if ks is None else np.asarray(ks)
        self.pose = dict(np.load(f"{D}/poses_{self.part}.npz")) if with_prev else None
        self.row2k = {int(r): k for k, r in enumerate(self.rows)}

    def __len__(self):
        return len(self.ks)

    def _open(self):
        if self.mm is not None:
            return
        self.mm = [np.load(f"{D}/feats/{self.part}_{g}.npy", mmap_mode="r") for g in ("dec32", "logit7")]
        self.gd = np.load(f"{DIST_DIR}/{self.part}_gt_dist_exact.npy", mmap_mode="r")
        self.gdir = np.load(f"{D}/dir_{self.part}.npy", mmap_mode="r")
        if self.split == "train":
            self.oof, self.row2 = {}, {}
            for f in range(3):
                d = f"{OUT}/e37_fold{f}"
                ids = np.load(f"{d}/ids_oof.npy")
                self.oof[f] = (np.load(f"{d}/prob_oof.npy", mmap_mode="r"), np.load(f"{d}/dir_oof.npy", mmap_mode="r"))
                for k, r in enumerate(ids):
                    self.row2[int(r)] = (f, k)

    def _x_of(self, k):
        row = int(self.rows[k])
        x = np.concatenate([np.asarray(m[row], np.float32) for m in self.mm], 0)
        x = (x - self.mean[:, None, None]) / self.std[:, None, None]
        if self.split == "train":
            f, kk = self.row2[row]
            heat, dirf = np.asarray(self.oof[f][0][kk], np.float32), np.asarray(self.oof[f][1][kk], np.float32)
        else:
            heat, dirf = np.asarray(self.es.prob[k], np.float32), np.asarray(self.es.dirf[k], np.float32)
        return np.concatenate([x, heat[None], dirf], 0).astype(np.float16)

    def __getitem__(self, i):
        self._open()
        k = int(self.ks[i])
        row = int(self.rows[k])
        x = self._x_of(k)
        o = dict(x=torch.from_numpy(x), gd=torch.from_numpy(np.asarray(self.gd[row], np.float32) * 0.5), gdir=torch.from_numpy(np.asarray(self.gdir[row], np.float16)), k=k, row=row)
        t = gt_tensors(self.gt[row])
        o.update({a: torch.from_numpy(b) for a, b in t.items()})
        if self.with_prev:  # previous frame of the same scene (2 Hz) and the ego poses of both frames: T.2 time recursion
            P = self.pose
            prow = int(P["prev"][row])
            pk = self.row2k.get(prow, -1) if prow >= 0 else -1
            o["has_prev"] = torch.tensor(1.0 if pk >= 0 else 0.0)
            o["x_prev"] = torch.from_numpy(self._x_of(pk)) if pk >= 0 else torch.zeros_like(o["x"])
            o["Rc"], o["tc"] = torch.from_numpy(P["R"][row].astype(np.float32)), torch.from_numpy(P["t"][row].astype(np.float32))
            o["Rp"], o["tp"] = (torch.from_numpy(P["R"][prow].astype(np.float32)), torch.from_numpy(P["t"][prow].astype(np.float32))) if pk >= 0 else (o["Rc"], o["tc"])
        return o


def collate(b):
    B = len(b)
    Gm = max(1, max(len(o["g05"]) for o in b))
    Hm = max(1, max(len(o["g1"]) for o in b))
    Js = max(1, max(len(o["js"]) for o in b))
    Jm = max(1, max(len(o["jm"]) for o in b))
    out = dict(x=torch.stack([o["x"] for o in b]), gd=torch.stack([o["gd"] for o in b]), gdir=torch.stack([o["gdir"] for o in b]), k=torch.tensor([o["k"] for o in b]), row=torch.tensor([o["row"] for o in b]),
               g05=torch.full((B, Gm, 2), 1e4), t05=torch.zeros(B, Gm, 2), n05=torch.zeros(B, Gm, dtype=torch.long), m05=torch.zeros(B, Gm),
               g1=torch.full((B, Hm, 2), 1e4), m1=torch.zeros(B, Hm), dg=torch.full((B, Hm, Hm), UNREACH, dtype=torch.int32),
               js=torch.full((B, Js, 2), 1e4), jm=torch.full((B, Jm, 2), 1e4))
    if "x_prev" in b[0]:
        for key in ("has_prev", "x_prev", "Rc", "tc", "Rp", "tp"):
            out[key] = torch.stack([o[key] for o in b])
    for i, o in enumerate(b):
        n, h = len(o["g05"]), len(o["g1"])
        if len(o["js"]):
            out["js"][i, : len(o["js"])] = o["js"]
        if len(o["jm"]):
            out["jm"][i, : len(o["jm"])] = o["jm"]
        if n:
            out["g05"][i, :n], out["t05"][i, :n], out["n05"][i, :n], out["m05"][i, :n] = o["g05"], o["t05"], o["n05"], 1
            out["g1"][i, :h], out["m1"][i, :h] = o["g1"], 1
            out["dg"][i, :h, :h] = o["dg"].to(torch.int32)
    return out


def loader(split="train", bs=8, workers=5, shuffle=True, ks=None, drop_last=True, with_prev=False):
    ds = BuilderData(split, ks=ks, train=(split == "train"), with_prev=with_prev)
    return ds, torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers, collate_fn=collate, drop_last=drop_last, persistent_workers=workers > 0, prefetch_factor=3 if workers > 0 else None)
