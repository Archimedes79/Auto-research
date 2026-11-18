"""GPU-resident loader of the E365 predicted-node cache (the training fields of one split on the device, batches assembled there): the same batches as `lane_ptok_data.loader`, no worker process, no host copy.

    ds, dl = loader("train", seed=0, bs=8, ident="e292_a", K=None, cache=LC.CACHE, dev="cuda")        # the signature of `lane_ptok_data.loader` (workers is accepted and ignored); dl yields batch dicts on `dev`
    batch keys: tok (B, K, 64) f16, pos, dirv, e, e_raw, s, uout, uin, flags, valid, ytype (int64), y (B, K, K) uint8, ylong (B, K, K) bool = y and the token chord > 8 m (a statistic of the loss), gt_out, token, row,
    and the fixed-shape pair list of the handle-scale term `sp_b`, `sp_i`, `sp_j` (B * P,), `sp_y` (B * P, 2), `sp_ok` (B * P,) (`jtoken_fast.fast_loss` reads it; no dense yscale is built unless dense_scale=True)

What is the same as the CPU loader: the fields and dtypes, the dense targets (`densify`: a pair that two links claim keeps the LAST one, here deterministically), the ORDER of the frames: with `shuffle` the epoch's
permutation is drawn exactly as torch's `DataLoader(shuffle=True, drop_last=True)` draws it (the same two numbers from the global torch generator per pass), so a run with the same seed sees the same batches in the
same order as before (tested against a real DataLoader). What differs: `tokens_np` and `label` are not built (a read-out needs them, the trainer does not), `token_drop` is not supported (use the trainer's per-step drop).
The device holds the TRAIN fields only (about 0.9 GB for the 8414 frames of K = 512: the pair lists are cut to the columns in use); the loader reads them once from the memory-mapped cache. Tests: grid/test_lane_ptok_gpu.py
"""
import os
import sys

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub_ in ("training/data", "training/grid", "evaluation"):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import lane_ptok_data as LP  # noqa: E402
import lane_tokcache as LC  # noqa: E402
import qtargets as QT  # noqa: E402

FIELDS = LP.TOKEN_FIELDS                                                     # tok pos dirv e e_raw s uout uin flags valid
SPARSE = ("ypair", "ypair_long", "ypair_scale")


class DeviceTokData:
    """the cache of one split on the device; `batch(idx)` assembles a batch of frames (idx: a list or a tensor of frame numbers)"""

    def __init__(self, split="train", seed=0, ident="e292_a", K=None, cache=LC.CACHE, dev="cuda", dense_scale=False, dir=None):
        d = dir or LP.cache_dir(split, seed, ident, K, cache)
        c = LC.TokCache(d)
        assert c.done == len(c), f"{d} is not finished ({c.done} of {len(c)} frames)"
        self.dir, self.K, self.rows, self.tokens, self.dev, self.dense_scale = d, c.K, c.rows, c.tokens, dev, dense_scale
        f = c.f
        t = lambda n: torch.from_numpy(np.ascontiguousarray(np.array(f[n]))).to(dev)  # noqa: E731
        self.f = {n: t(n) for n in FIELDS if n in f}
        self.gt_type, self.gt_out = t("gt_type"), t("gt_out")
        yp = np.array(f["ypair"])
        used = np.flatnonzero((yp[..., 0] >= 0).any(0))
        self.P = int(used.max()) + 1 if len(used) else 1                      # the pair columns in use (one past the last used slot, front-packed or not): the fixed width of `sp_*`; the stored width (4096) is larger
        self.ypair = torch.from_numpy(np.ascontiguousarray(yp[:, :self.P])).to(dev)         # int16 on the device, cast per batch
        self.ypl = torch.from_numpy(np.ascontiguousarray(np.array(f["ypair_long"])[:, :self.P])).to(dev)
        self.yps = torch.from_numpy(np.ascontiguousarray(np.array(f["ypair_scale"])[:, :self.P])).to(dev)
        self.rows_np = np.asarray(self.rows)
        self.n = len(self.rows)

    def __len__(self):
        return self.n

    def batch(self, idx):
        idx_cpu = np.asarray(idx.cpu() if torch.is_tensor(idx) else idx, dtype=np.int64)       # a list (the loader) costs no device sync
        idx = torch.as_tensor(idx_cpu, device=self.dev)
        B, K, P = len(idx), self.K, self.P
        out = {n: v[idx] for n, v in self.f.items()}
        out["valid"] = out["valid"].bool()
        out["ytype"] = self.gt_type[idx].long()
        yp, ypl, yps = self.ypair[idx].long(), self.ypl[idx], self.yps[idx]
        ok = yp[..., 0] >= 0
        i, j = yp[..., 0].clamp_min(0), yp[..., 1].clamp_min(0)
        bi = torch.arange(B, device=self.dev)[:, None].expand_as(ok)
        key = (bi * K + i) * K + j                                              # a cell of the (B, K, K) maps
        order = torch.arange(P, device=self.dev)[None].expand_as(ok)
        dump = B * K * K                                                        # one extra cell takes the padding rows: no boolean-mask indexing, no host sync
        keyd = torch.where(ok, key, torch.full_like(key, dump))
        last = torch.full((dump + 1,), -1, dtype=torch.long, device=self.dev)
        last.scatter_reduce_(0, keyd.reshape(-1), torch.where(ok, order, torch.full_like(order, -1)).reshape(-1), "amax", include_self=True)    # the LAST pair of a cell wins (numpy's assignment order), deterministically
        win = ok & (last[keyd] == order)
        y = torch.zeros(dump + 1, dtype=torch.uint8, device=self.dev)
        y[keyd.reshape(-1)] = 1                                                 # the same value for every claim of a cell
        y = y[:dump].view(B, K, K)
        pos = out["pos"].double()
        chord = (pos[:, None, :, :] - pos[:, :, None, :]).norm(dim=-1)
        out["y"], out["ylong"] = y, (y > 0) & (chord > QT.SHORT)
        out["gt_out"], out["token"], out["row"] = self.gt_out[idx], [self.tokens[int(k)] for k in idx_cpu], torch.as_tensor(self.rows_np[idx_cpu], device=self.dev)
        w2 = win[..., None]
        sp_y = torch.where(w2, yps, torch.full_like(yps, float("nan")))
        out.update(sp_b=bi.reshape(-1), sp_i=i.reshape(-1), sp_j=j.reshape(-1), sp_y=sp_y.reshape(-1, 2), sp_ok=(win & torch.isfinite(yps[..., 0])).reshape(-1))
        if self.dense_scale:                                                    # for the parity tests and the plain loss only: the dense (B, K, K, 2) map (16 MB per batch of 8)
            ys = torch.full((dump + 1, 2), float("nan"), device=self.dev)
            ys[torch.where(win, key, torch.full_like(key, dump)).reshape(-1)] = yps.reshape(-1, 2)         # unique cells for the winners; the losers write the dump cell
            out["yscale"] = ys[:dump].view(B, K, K, 2)
        return out


class DeviceLoader:
    """pass after pass over the frames in the order of `DataLoader(shuffle=True, drop_last=True)` (shuffle) or in file order (no shuffle, the last batch kept); yields `DeviceTokData.batch`"""

    def __init__(self, ds, bs, shuffle, persistent=False):
        self.ds, self.bs, self.shuffle, self.persistent, self.passes = ds, bs, shuffle, persistent, 0     # persistent: the order of a DataLoader with persistent workers (the base seed is drawn once, at its first pass)

    def __len__(self):
        return len(self.ds) // self.bs if self.shuffle else -(-len(self.ds) // self.bs)

    def __iter__(self):
        n = len(self.ds)
        first, self.passes = self.passes == 0, self.passes + 1
        if self.shuffle:
            if first or not self.persistent:
                torch.empty((), dtype=torch.int64).random_()                     # the DataLoader's base seed (drawn when its iterator is made: every pass, or once with persistent workers)
            gen = torch.Generator()
            gen.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))      # the RandomSampler's seed
            order = torch.randperm(n, generator=gen).tolist()
        else:
            order = list(range(n))
        for s in range(0, len(self) * self.bs, self.bs):
            yield self.ds.batch(order[s:s + self.bs])


def loader(split="train", seed=0, bs=8, workers=0, token_drop=0.0, ident="e292_a", K=None, shuffle=None, cache=LC.CACHE, dir=None, dev=None, dense_scale=False):
    """the signature of `lane_ptok_data.loader` (workers only selects the order of a DataLoader with workers (persistent) or without: nothing runs in a worker) -> (DeviceTokData, DeviceLoader)"""
    assert not token_drop, "token_drop is the trainer's per-step drop here (jtoken_train --token-drop), not the loader's"
    shuffle = split == "train" if shuffle is None else shuffle
    ds = DeviceTokData(split, seed, ident, K, cache, dev or ("cuda" if torch.cuda.is_available() else "cpu"), dense_scale, dir)
    return ds, DeviceLoader(ds, bs, shuffle, persistent=workers > 0)
