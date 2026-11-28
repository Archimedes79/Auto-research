"""E357 data: the 39 frozen BEV channels of a frame + its GT node table on a lattice (the tokens of E357's pair head; today read by the token cache and the live tokens) and the GT links, padded and masked.

    ds, dl = loader("train" | "val", res=2.5, bs=8, workers=4)
    python -P -m lane_center.net.prep.lane_tok_data build-cache --split train --lattice 2.5 [--max N] [--threads 2]      # the cache of the node tables (one process; frames already there are skipped)

Frames: TRAIN = the train rows of `lane_gt_data.QData` with a stored lane GT; VAL = exactly the rows of data/splits.json['val'] in that order (asserted: 873, none missing; the val feature files also hold the TEST rows, never read).
The features and the token of a row are read here as QData reads them (its item path builds 2 m targets with sizes bound at import and must not run on another lattice).
Table of a frame (`frame_table`, numpy; float64 so that a read-out of the GT links is the oracle's to the last bit): pos, dirv (M, 2), cell (M, 2), flags (M, 8), piece (M,), sstart (M,) of `qtargets.node_table` on the lattice;
links src, dst (L,), long (L,) (chord > 8 m), scale (L, 2) the fitted handle scales of the long links (NaN for short ones, `qtargets.fit_long_scales`, about 0.1 s per frame: the reason for the cache), conn (L,) = more than half
of the lane GT points on the link's path are connector points (the link classes of E346: junction link = long and conn). flags / piece / sstart / cell / conn are for the read-out and the tables only: no model input reads them.
Cache: runs/e357_cache/l<lattice>_<lane GT signature>_<TABLE_VERSION>/<split>/<token>.npz (derived data; the version is bumped by hand when the table code changes; a file is written whole and renamed; a table missing in the cache is computed and stored by whoever asks, also a loader worker).
Batch (`collate`): x (B, 39, 200, 200), token, row, pos / dirv (B, N, 2) float32 and valid (B, N) with N = the most nodes of a frame of the batch, y (B, N, N) = 1 where i -> j is a GT link, ys (B, N, N, 2) the fitted scales
of the long GT links (NaN elsewhere), table = the per-frame tables. Tests: through net/tests/test_lane_tokcache.py and test_lane_net.py (net/tests/test_jtok.py is in the tag v1)
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from lane_center.training import bflc_paths as BP  # where code, data and runs are
import GtGenerator  # the ground truth: its public interface only
from lane_center.net.prep import lane_gt_data as LD
from lane_center.net.prep import qtargets as QT

CACHE = os.path.join(BP.RUNS, "e357_cache")
TABLE_VERSION = "t1"                                                        # part of the cache key: bump it BY HAND whenever `frame_table` (or what it reads of qtargets) changes what a table holds
TABLE = ("pos", "dirv", "cell", "flags", "piece", "sstart", "src", "dst", "long", "scale", "conn")


def connector_points(g):
    """(P,) bool over the points of all pieces in order (the order of `node_table`'s points): a `connector` segment of the lane GT covers the point"""
    out = []
    for p in g["pieces"]:
        k = np.zeros(len(p["pts"]), bool)
        for s in p["segments"]:
            if s["kind"] == "connector":
                k[s["i0"]: s["i1"] + 1] = True
        out.append(k)
    return np.concatenate(out) if out else np.zeros(0, bool)


def frame_table(g, res):
    """the table of a lane GT graph on the `res` m lattice (module docstring); the lattice is in force only inside this call and is checked in the calling process (main or worker)"""
    with QT.lattice(res):
        QT.check_lattice(res)
        nt = QT.node_table(g)
    gk = connector_points(g)
    return dict(pos=nt["pos"].astype(np.float64), dirv=nt["dirv"].astype(np.float64), cell=nt["cell"].astype(np.int16), flags=nt["flags"].astype(bool), piece=nt["piece"].astype(np.int32), sstart=nt["sstart"].astype(bool),
                src=nt["src"].astype(np.int32), dst=nt["dst"].astype(np.int32), long=nt["long"].astype(bool), scale=QT.fit_long_scales(nt).astype(np.float64).reshape(-1, 2),
                conn=np.array([gk[p].mean() > 0.5 for p in nt["path"]], bool))


def _save(path, t):
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    np.savez(tmp, **t)
    os.replace(tmp, path)


class TokData(torch.utils.data.Dataset):
    def __init__(self, split="train", res=2.5, cache=CACHE, feats=True):
        assert split in ("train", "val"), split
        assert (QT.N, QT.RES) == (50, 2.0), "qtargets is not on its 2 m defaults outside a lattice block"
        self.qd = LD.QData("train" if split == "train" else "val_clean", map_mode="error", pixels=False)
        self.split, self.res, self.feats = split, float(res), feats
        self.rows = [int(self.qd.bd.rows[k]) for k in self.qd.ks]
        self.tokens = [self.qd.tokens[r] for r in self.rows]
        if split == "val":
            want = [int(r) for r in json.load(open(os.path.join(BP.ROOT, "data", "splits.json")))["val"]]
            assert self.qd.missing == 0 and len(want) == 873 and self.rows == want, f"VAL is not the 873 rows of data/splits.json['val'] ({len(self.rows)} rows, {self.qd.missing} without lane GT)"
        self.dir = None if cache is None else os.path.join(cache, f"l{self.res:g}_{GtGenerator.lane_signature()}_{TABLE_VERSION}", split)
        if self.dir:
            os.makedirs(self.dir, exist_ok=True)

    def __len__(self):
        return len(self.rows)

    def cached(self):
        """how many frames of the split have their table in the cache"""
        have = set(os.listdir(self.dir)) if self.dir else set()
        return sum(t + ".npz" in have for t in self.tokens)

    def table(self, i):
        path = os.path.join(self.dir, self.tokens[i] + ".npz") if self.dir else None
        if path and os.path.exists(path):
            with np.load(path) as z:
                t = {k: z[k] for k in TABLE}
            assert np.array_equal(t["cell"], np.clip(np.floor((t["pos"] + QT.HALF) / self.res), 0, round(2 * QT.HALF / self.res) - 1)), f"{path} is not a table of the {self.res} m lattice"
            return t
        t = frame_table(GtGenerator.load_lane(self.tokens[i]), self.res)
        if path:
            _save(path, t)
        return t

    def __getitem__(self, i):
        out = dict(self.table(i), token=self.tokens[i], row=self.rows[i])
        if self.feats:                                                                # as `lane_gt_data.QData.__getitem__`: the 39 channels, normalised, float16
            self.qd._open()
            bd = self.qd.bd
            x = np.concatenate([np.asarray(m[self.rows[i]], np.float32) for m in self.qd.mm], 0)
            out["x"] = torch.from_numpy(((x - bd.mean[:39, None, None]) / bd.std[:39, None, None]).astype(np.float16))
        return out


# ------------------------------------------------------------------------------------------------------------------------------ the table of a node subset
def sub_table(t, keep):
    """the table of the kept nodes. Links: a GT link between two kept nodes stays (with its scale and class); a chain of GT links through removed nodes becomes ONE link between its kept ends (conn = a link of the
    chain is on a connector path, no fitted scale), unless the direct link exists"""
    keep = np.asarray(keep, bool)
    new = np.cumsum(keep) - 1
    outs = [[] for _ in keep]
    for l, s in enumerate(t["src"]):
        outs[s].append(l)
    links = {}                                                              # (a, b) -> (conn, scale)
    for l, (a, b) in enumerate(zip(t["src"], t["dst"])):
        if keep[a] and keep[b] and a != b:
            links[(int(a), int(b))] = (bool(t["conn"][l]), t["scale"][l])
    for a in np.flatnonzero(keep):
        stack, seen = [(l, bool(t["conn"][l])) for l in outs[a] if not keep[t["dst"][l]]], set()
        while stack:
            l, conn = stack.pop()
            b = int(t["dst"][l])
            if keep[b]:
                if b != a:
                    links.setdefault((int(a), b), (conn, np.full(2, np.nan)))
            elif b not in seen:
                seen.add(b)
                stack += [(l2, conn or bool(t["conn"][l2])) for l2 in outs[b]]
    src, dst = (np.array([k[i] for k in links], int) for i in (0, 1))
    out = {k: t[k][keep] for k in ("pos", "dirv", "cell", "flags", "piece", "sstart")}
    out.update(src=new[src].astype(np.int32), dst=new[dst].astype(np.int32), long=np.linalg.norm(t["pos"][src] - t["pos"][dst], axis=1) > QT.SHORT,
               scale=np.array([v[1] for v in links.values()], float).reshape(-1, 2), conn=np.array([v[0] for v in links.values()], bool))
    return out


def collate(b):
    B, N = len(b), max(1, max(len(o["pos"]) for o in b))
    out = dict(token=[o["token"] for o in b], row=torch.tensor([o["row"] for o in b]), table=[{k: o[k] for k in TABLE} for o in b])
    if "x" in b[0]:
        out["x"] = torch.stack([o["x"] for o in b])
    pos, dirv, valid = torch.zeros(B, N, 2), torch.zeros(B, N, 2), torch.zeros(B, N, dtype=torch.bool)
    y, ys = torch.zeros(B, N, N), torch.full((B, N, N, 2), float("nan"))
    for i, o in enumerate(b):
        n = len(o["pos"])
        pos[i, :n], dirv[i, :n], valid[i, :n] = torch.from_numpy(o["pos"]).float(), torch.from_numpy(o["dirv"]).float(), True
        s, d, lg = torch.from_numpy(o["src"]).long(), torch.from_numpy(o["dst"]).long(), torch.from_numpy(o["long"])
        y[i, s, d] = 1.0
        ys[i, s[lg], d[lg]] = torch.from_numpy(o["scale"]).float()[lg]
    out.update(pos=pos, dirv=dirv, valid=valid, y=y, ys=ys)
    return out


def loader(split="train", res=2.5, bs=8, workers=4, shuffle=None, cache=CACHE, feats=True):
    shuffle = split == "train" if shuffle is None else shuffle
    ds = TokData(split, res, cache=cache, feats=feats)
    return ds, torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers, collate_fn=collate, drop_last=shuffle, persistent_workers=workers > 0, prefetch_factor=2 if workers > 0 else None)


def build_cache(split, res, n=None, cache=CACHE, log=print):
    """the tables of the first `n` frames (all) of the split into the cache, one process -> (frames asked for, tables computed now)"""
    ds = TokData(split, res, cache=cache, feats=False)
    n, have, t0, new = min(n or len(ds), len(ds)), set(os.listdir(ds.dir)), time.time(), 0
    for i in range(n):
        if ds.tokens[i] + ".npz" not in have:
            ds.table(i)
            new += 1
        if log and ((i + 1) % 500 == 0 or i + 1 == n):
            log(f"   {split} l{res:g}: {i + 1}/{n} frames, {new} computed, {time.time() - t0:.0f} s ({(time.time() - t0) / max(new, 1):.3f} s per computed frame)", flush=True)
    return n, new


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("build-cache")
    s.add_argument("--split", required=True, choices=("train", "val"))
    s.add_argument("--lattice", type=float, default=2.5)
    s.add_argument("--max", type=int, default=None, help="only the first N frames of the split")
    s.add_argument("--threads", type=int, default=2, help="thread limit of this (single) process")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    n_, new_ = build_cache(a.split, a.lattice, a.max)
    print(f"cache {os.path.join(CACHE, '')}: {n_} frames of {a.split} at {a.lattice:g} m, {new_} computed now")
