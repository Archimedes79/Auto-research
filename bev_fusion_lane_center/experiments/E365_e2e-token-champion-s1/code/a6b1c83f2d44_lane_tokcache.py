"""D113 predicted-node cache: the tokens of a FROZEN lane head B (a `QGraph` run, e.g. E292 `runs/e292_a_s<S>` = B' with the Matrix-NMS layer, or E273 `runs/e273_p_s<S>`) on every frame of TRAIN and VAL,
with the GT labels of every token. Derived data (not in git; this tool, its selftest and the manifest are).

    wsl bash tools/env/_pyr.sh lane_tokcache.py build   --run runs/e292_a_s0 --split val|train [--K 320] [--no-f2] [--r 1.5] [--limit N]      # GPU, resumes; one directory per run and split
    wsl bash tools/env/_py14.sh lane_tokcache.py fixture --cache <dir> --dump temp/e292/predfull_e292_a_s0.pkl                                    # the cache against B's stored dump, bit for bit (VAL)
    wsl bash tools/env/_py14.sh lane_tokcache.py report  --cache <dir>                                                                            # ceilings, delivered-node cut, recall / precision at r
    wsl bash tools/env/_py14.sh lane_tokcache.py selftest
    c = TokCache(dir); f = c[i]                                                                                                                  # one frame as numpy arrays; c.reassign(i, r) re-matches the tokens at another radius

Frames: the order of `lane_tok_data.TokData(split, res=2.0)`: TRAIN = the rows of `QData('train')` with a stored lane GT, VAL = exactly the 873 rows of data/splits.json['val'] (asserted by TokData).
Tokens of a frame: the K cells with the largest existence logit `e` of the node stage (`QGraph.node_stage`: for B' after the Matrix-NMS layer, `e_raw` = the head's own logit), a stable sort, ties -> the lower flat cell;
`valid` = sigmoid(e) > 0.05 (B's own active threshold), the other rows are padding. Per token (all float32 unless said): cell (int16, flat row * 50 + col, row = x, col = y), e, e_raw, e_p = sigmoid(e) as the GPU computes it,
pos (2, metres), off (2), dirv (2, unit), dir_raw (2), flags (8 logits), flags_p (8 probabilities), s, s_p, uout, uin, tok (float16, 64: the refined node token that B's link head reads), f2 (float16, 144: the 2 m pyramid
feature of the cell; optional). B is frozen: nothing here carries a gradient; the logits are stored so that a head can use them as inputs.
Labels: gt_id (int16, the row of the frame's GT node table `TokData.table(i)` of the token, -1 none) by a UNIQUE assignment: all (valid token, GT node) pairs closer than r metres, taken in order of distance (ties: the
higher-ranked token, then the lower GT node), a token and a GT node are used once (a greedy closest-first rule, no optimisation: the closest pairs win); gt_dist (metres, NaN none); gt_type (int8, `TYPES`: 0 none, 1 lane point, 2 fork,
3 merge, 4 entry (stop), 5 exit, 6 boundary (start, end, win_in, win_out); several flags: the first of fork, merge, entry, exit, boundary), gt_flags (8 bool, the matched node's flags in the order of `qtargets.FLAGS`), gt_aux (3 bool:
section start, start of a long link, end of a long link). GT links on tokens (padded to LMAX): link_tok (token i, token j; -1 pad) where both ends have a token, link_long, link_conn (the junction link class of E346), link_scale
(fitted handle scales of the long links, NaN for short ones); n_gt, n_gt_matched, n_links (all GT links), n_lost (links with an end without a token); n_e50 / n_e20 (cells of the whole map with sigmoid(e) >= 0.5 / 0.2),
match_at / lost_at (F, len(KS)): matched GT nodes and lost links when only the first KS[j] tokens are kept (the ceilings).
Directory: runs/tokcache/<run name>_K<K>/<split>/ with one .npy per field, frames.json, progress.json, manifest.json. A build is resumable (progress.json counts complete batches of 4; the arrays are memory maps).
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub_ in ("training/data", "training/grid", "evaluation"):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))

VERSION = "v1"
CACHE = os.path.join(ROOT, "runs", "tokcache")
TYPES = ("none", "lane", "fork", "merge", "entry", "exit", "boundary")
FLAGS = ("stop", "exit", "split", "merge", "start", "end", "win_in", "win_out")                  # = qtargets.FLAGS (asserted in `build`)
KS = (128, 192, 256, 320, 384, 512)                                                              # token budgets of the ceiling statistics
N, RES, HALF = 50, 2.0, 50.0
P_ACTIVE = 0.05                                                                                    # B's own active threshold (qlinks.LinkHead: valid = sigmoid(e) > thr)
LMAX = 512


# ----------------------------------------------------------------------------------------------------------- pure parts (selftest)
def select_tokens(no, f2_map, K):
    """node-stage outputs `no` (the dict of `QGraph.node_stage`, a batch) and the 2 m pyramid map `f2_map` (B, C, 50, 50) or None -> dict of tensors (B, K, ...) of the fields in the module docstring; the probabilities are made on the
    whole maps first and gathered afterwards, exactly as `qpredict.predict_from_out` does, so that they are the same numbers"""
    e = no["e"].float().flatten(1)
    B = e.shape[0]
    order = torch.argsort(e, dim=1, descending=True, stable=True)[:, :K]                                     # equal logits keep their cell order: the lower flat cell first

    def g1(t):                                                                                                 # (B, 50, 50) or (B, 2500) -> (B, K)
        return t.flatten(1).gather(1, order)

    def gc(t):                                                                                                 # (B, C, 50, 50) -> (B, K, C)
        return t.flatten(2).gather(2, order[:, None, :].expand(-1, t.shape[1], -1)).transpose(1, 2)

    ep = e.sigmoid()
    dr = gc(no["dir_raw"].float())
    out = dict(cell=order.to(torch.int16), e=g1(e), e_raw=g1(no.get("e_raw", no["e"]).float()), e_p=g1(ep), valid=g1(ep) > P_ACTIVE, pos=gc(no["pos"].float()), off=gc(no["off"].float()),
               dirv=dr / dr.norm(dim=-1, keepdim=True).clamp_min(1e-6), dir_raw=dr, flags=gc(no["flags"].float()), flags_p=gc(no["flags"].float().sigmoid()), s=g1(no["s"].float()), s_p=g1(no["s"].float().sigmoid()),
               uout=g1(no["uout"].float()), uin=g1(no["uin"].float()), tok=gc(no["tok"]).to(torch.float16), n_e50=(ep >= 0.5).sum(1), n_e20=(ep >= 0.2).sum(1))
    if f2_map is not None:
        out["f2"] = gc(f2_map).to(torch.float16)
    return out


def assign_unique(tpos, tvalid, gpos, r):
    """valid tokens (T, 2) / (T,) and GT nodes (M, 2) -> (gt_id (T,) int, dist (T,) float): the pairs closer than r (<=), closest first, ties: the lower token (= the higher existence), then the lower GT node; each used once"""
    T, M = len(tpos), len(gpos)
    gt, dist = -np.ones(T, int), np.full(T, np.nan)
    if T == 0 or M == 0:
        return gt, dist
    d = np.linalg.norm(np.asarray(tpos, float)[:, None, :] - np.asarray(gpos, float)[None, :, :], axis=2)
    d[~np.asarray(tvalid, bool)] = np.inf
    ti, gi = np.nonzero(d <= r)
    if len(ti) == 0:
        return gt, dist
    o = np.lexsort((gi, ti, d[ti, gi]))
    used = np.zeros(M, bool)
    for k in o:
        t, g = ti[k], gi[k]
        if gt[t] < 0 and not used[g]:
            gt[t], dist[t], used[g] = g, d[t, g], True
    return gt, dist


def types_of(flags):
    """GT flags (n, 8) bool in the order of `FLAGS` -> (n,) int8 of `TYPES` for matched nodes (1 lane point when no flag): split -> fork, merge, stop -> entry, exit, start / end / win_in / win_out -> boundary"""
    f = np.asarray(flags, bool).reshape(-1, 8)
    t = np.ones(len(f), np.int8)
    for code, cols in ((6, (4, 5, 6, 7)), (5, (1,)), (4, (0,)), (3, (3,)), (2, (2,))):                       # lowest priority first, later rows overwrite
        t[f[:, list(cols)].any(1)] = code
    return t


def label_frame(tpos, tvalid, table, r, K=None, ks=KS):
    """the labels of one frame: tokens (T, 2) / (T,) in rank order and the GT table of `lane_tok_data` (pos, flags, sstart, src, dst, long, conn, scale) -> dict of the label fields (module docstring) for the first K tokens (default all);
    the ceiling statistics `match_at` / `lost_at` are made for the first ks[j] of ALL the tokens given (so T may be larger than K)"""
    K, M = (len(tpos) if K is None else K), len(table["pos"])
    tpos_all, tvalid_all = tpos, tvalid
    tpos, tvalid = tpos[:K], tvalid[:K]
    gt, dist = assign_unique(tpos, tvalid, table["pos"], r)
    got = gt >= 0
    gflags = np.zeros((K, 8), bool)
    gaux = np.zeros((K, 3), bool)
    long_ = np.asarray(table["long"], bool)
    src, dst = np.asarray(table["src"], int), np.asarray(table["dst"], int)
    uout, uin = np.zeros(M, bool), np.zeros(M, bool)
    uout[src[long_]], uin[dst[long_]] = True, True
    if got.any():
        gflags[got] = np.asarray(table["flags"], bool)[gt[got]]
        gaux[got] = np.stack([np.asarray(table["sstart"], bool)[gt[got]], uout[gt[got]], uin[gt[got]]], 1)
    gtype = np.zeros(K, np.int8)
    gtype[got] = types_of(gflags[got])

    def links(gt_):
        tok_of = -np.ones(M, int)
        tok_of[gt_[gt_ >= 0]] = np.nonzero(gt_ >= 0)[0]
        ti, tj = tok_of[src], tok_of[dst]
        return ti, tj, (ti >= 0) & (tj >= 0)

    ti, tj, ok = links(gt)
    nl = int(ok.sum())
    assert nl <= LMAX, f"{nl} GT links on tokens exceed LMAX {LMAX}"
    lt = -np.ones((LMAX, 2), np.int16)
    lt[:nl] = np.stack([ti[ok], tj[ok]], 1)
    ll, lc = np.zeros(LMAX, bool), np.zeros(LMAX, bool)
    ls = np.full((LMAX, 2), np.nan, np.float32)
    ll[:nl], lc[:nl] = long_[ok], np.asarray(table["conn"], bool)[ok]
    ls[:nl] = np.asarray(table["scale"], np.float32).reshape(-1, 2)[ok]
    match_at, lost_at = np.zeros(len(ks), np.int16), np.zeros(len(ks), np.int16)
    for j, k in enumerate(ks):
        g2, _ = assign_unique(tpos_all[:k], tvalid_all[:k], table["pos"], r)
        _, _, ok2 = links(g2)
        match_at[j], lost_at[j] = int((g2 >= 0).sum()), int((~ok2).sum())
    return dict(gt_id=gt.astype(np.int16), gt_dist=dist.astype(np.float32), gt_type=gtype, gt_flags=gflags, gt_aux=gaux, link_tok=lt, link_long=ll, link_conn=lc, link_scale=ls,
                n_gt=np.int16(M), n_gt_matched=np.int16(got.sum()), n_links=np.int16(len(src)), n_lost=np.int16(len(src) - nl), match_at=match_at, lost_at=lost_at)


def compare_dump(cache, dump, e_min=0.2, prefix=False):
    """the cache against a stored B dump (`qgraph_readout.dump_full`: the nodes with sigmoid(e) >= e_min of every frame in the order of the active set): -> (ok, report dict). Per frame the first min(K, n) stored nodes are
    compared with the cache tokens cell by cell: same set of cells, and e, pos, dirv, flags, s equal bit for bit (the dump holds float64 copies of float32 numbers); the cache must hold no token at or above e_min behind them"""
    if prefix:                                                                                                  # a limited (smoke) cache: its frames are the first ones of the dump
        dump = dump[:len(cache.rows)]
    assert len(dump) == len(cache.rows), f"{len(dump)} dump frames, {len(cache.rows)} cache frames"
    worst, bad, cut, n_nodes, order_diff = dict(e=0.0, pos=0.0, dirv=0.0, flags=0.0, s=0.0), [], 0, 0, 0
    for i, d in enumerate(dump):
        if int(d["row"]) != int(cache.rows[i]):
            return False, dict(error=f"row of frame {i}: dump {d['row']}, cache {cache.rows[i]}")
        f = cache.f
        n = int(len(d["e"]))
        nc = min(n, cache.K)
        cut += int(n > cache.K)
        cell_d = np.clip(np.floor((np.asarray(d["pos"][:nc], float) + HALF) / RES).astype(int), 0, N - 1)
        cell_d = cell_d[:, 0] * N + cell_d[:, 1]
        cell_c = f["cell"][i, :nc].astype(int)
        order_diff += int(not np.array_equal(cell_d, cell_c))
        if sorted(cell_d.tolist()) != sorted(cell_c.tolist()):
            bad.append((i, "cells differ"))
            continue
        pc = {int(c): k for k, c in enumerate(cell_c)}
        for k in range(nc):
            j = pc[int(cell_d[k])]
            for name, a, b in (("e", d["e"][k], f["e_p"][i, j]), ("pos", d["pos"][k], f["pos"][i, j]), ("dirv", d["dirv"][k], f["dirv"][i, j]), ("flags", d["flags"][k], f["flags_p"][i, j]), ("s", d["s"][k], f["s_p"][i, j])):
                a64, b64 = np.asarray(a, np.float64), np.asarray(b, np.float32).astype(np.float64)
                v = float(np.abs(a64 - b64).max()) if a64.size else 0.0
                worst[name] = max(worst[name], v) if v == v else float("nan")
                if v != 0.0:
                    bad.append((i, name))
        if n < cache.K and nc == n:
            assert not (f["e_p"][i, n:] >= e_min).any(), f"frame {i}: a cache token at or above {e_min} behind the {n} stored nodes"
        n_nodes += nc
    ok = not bad and all(v == 0.0 for v in worst.values())
    return ok, dict(frames=len(dump), nodes_compared=n_nodes, frames_cut_by_K=cut, frames_with_other_order=order_diff, worst=worst, mismatches=len(bad), first=bad[:5])


# ----------------------------------------------------------------------------------------------------------- storage
FIELDS = dict(cell="int16", valid="bool", e="float32", e_raw="float32", e_p="float32", pos="float32", off="float32", dirv="float32", dir_raw="float32", flags="float32", flags_p="float32", s="float32", s_p="float32",
              uout="float32", uin="float32", tok="float16", f2="float16", gt_id="int16", gt_dist="float32", gt_type="int8", gt_flags="bool", gt_aux="bool", link_tok="int16", link_long="bool", link_conn="bool",
              link_scale="float32", n_gt="int16", n_gt_matched="int16", n_links="int16", n_lost="int16", n_e50="int16", n_e20="int16", match_at="int16", lost_at="int16")


def _path(d, name):
    return os.path.join(d, name + ".npy")


def _atomic_json(path, obj):
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


class TokCache:
    """read access to a cache directory: `c.rows`, `c.tokens`, `c.K`, `c.f` (the memory maps by field name), `c[i]` one frame as a dict of numpy arrays (copies), `c.reassign(i, r, table)` the labels at another radius"""

    def __init__(self, d):
        self.dir = d
        fr = json.load(open(os.path.join(d, "frames.json")))
        self.rows, self.tokens = fr["rows"], fr["tokens"]
        self.f = {n: np.load(_path(d, n), mmap_mode="r") for n in FIELDS if os.path.exists(_path(d, n))}
        self.K = self.f["cell"].shape[1]
        self.done = json.load(open(os.path.join(d, "progress.json")))["done"] if os.path.exists(os.path.join(d, "progress.json")) else 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        out = {n: np.array(a[i]) for n, a in self.f.items()}
        out.update(row=self.rows[i], token=self.tokens[i])
        return out

    def reassign(self, i, r, table):
        """the label fields of frame i for radius r from the stored positions and the frame's GT table (`lane_tok_data.TokData.table(i)`)"""
        return label_frame(np.array(self.f["pos"][i], float), np.array(self.f["valid"][i], bool), table, r)


def nodes_of(frame, sel=None):
    """the qformat 'nodes' dict (`qformat.py`) of one cached frame, the tokens of `sel` (default: the valid ones): pos, dirv, e (probability), flags (probabilities), s (probability), cell (row, col), section = the node's own index
    (a head that learns sections replaces it)"""
    sel = np.nonzero(frame["valid"])[0] if sel is None else np.asarray(sel)
    c = frame["cell"][sel].astype(int)
    return dict(pos=frame["pos"][sel].astype(float), dirv=frame["dirv"][sel].astype(float), e=frame["e_p"][sel].astype(float), flags=frame["flags_p"][sel].astype(float), s=frame["s_p"][sel].astype(float),
                cell=np.stack([c // N, c % N], 1), section=np.arange(len(sel)))


# ----------------------------------------------------------------------------------------------------------- build
def _create(d, n_frames, K, shapes, r, meta):
    os.makedirs(d, exist_ok=True)
    for name, shp in shapes.items():
        np.lib.format.open_memmap(_path(d, name), mode="w+", dtype=FIELDS[name], shape=(n_frames,) + shp).flush()
    _atomic_json(os.path.join(d, "meta.json"), dict(meta, K=K, r=r, version=VERSION, shapes={k: list(v) for k, v in shapes.items()}))


def _shapes(K, d_tok, d_f2):
    s = dict(cell=(K,), valid=(K,), e=(K,), e_raw=(K,), e_p=(K,), pos=(K, 2), off=(K, 2), dirv=(K, 2), dir_raw=(K, 2), flags=(K, 8), flags_p=(K, 8), s=(K,), s_p=(K,), uout=(K,), uin=(K,), tok=(K, d_tok),
             gt_id=(K,), gt_dist=(K,), gt_type=(K,), gt_flags=(K, 8), gt_aux=(K, 3), link_tok=(LMAX, 2), link_long=(LMAX,), link_conn=(LMAX,), link_scale=(LMAX, 2), n_gt=(), n_gt_matched=(), n_links=(), n_lost=(),
             n_e50=(), n_e20=(), match_at=(len(KS),), lost_at=(len(KS),))
    if d_f2:
        s["f2"] = (K, d_f2)
    return s


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()[:16]


def build(a):
    import lane_tok_data as LT
    import qpredict as QP
    import qtargets as QT
    assert tuple(QT.FLAGS) == FLAGS
    split = {"val": "val", "val_clean": "val", "train": "train"}[a.split]
    run = a.run if os.path.isabs(a.run) else os.path.join(ROOT, a.run)
    name = os.path.basename(os.path.normpath(run))
    out = os.path.join(CACHE, f"{name}_K{a.K}" + (f"_first{a.limit}" if a.limit else ""), split)                  # a limited build never shares a directory with the full one
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = LT.TokData(split, res=2.0, feats=True)
    n_all = len(ds) if not a.limit else min(a.limit, len(ds))
    m = QP.load_model(run, dev)
    bs, workers = 4, a.workers
    prog = os.path.join(out, "progress.json")
    done = json.load(open(prog))["done"] if os.path.exists(prog) else 0
    fresh = not os.path.exists(os.path.join(out, "meta.json"))
    assert done % bs == 0 and done <= n_all, f"progress {done} is not a multiple of {bs} within {n_all}"
    sub = torch.utils.data.Subset(ds, list(range(done, n_all)))
    dl = torch.utils.data.DataLoader(sub, batch_size=bs, shuffle=False, num_workers=workers, collate_fn=_collate, persistent_workers=workers > 0, prefetch_factor=2 if workers > 0 else None)
    arrays, t0, nb = None, time.time(), 0
    os.makedirs(out, exist_ok=True)
    _atomic_json(os.path.join(out, "frames.json"), dict(rows=ds.rows[:n_all], tokens=ds.tokens[:n_all]))
    i = done
    for b in dl:
        with torch.no_grad():
            x = b["x"].to(dev).float()
            P = m.pyr(x)
            no = m.node_stage(P)
            tk = select_tokens(no, P["feat2"] if a.f2 else None, max(a.K, max(KS)))                              # all the tokens the ceiling statistics need, K of them are stored
        tk = {k: v.cpu().numpy() for k, v in tk.items()}
        pos_all, valid_all = tk["pos"], tk["valid"]
        if arrays is None:
            if fresh:
                _create(out, n_all, a.K, _shapes(a.K, tk["tok"].shape[-1], tk["f2"].shape[-1] if "f2" in tk else 0), a.r,
                        dict(run=a.run, run_sha=_sha(os.path.join(run, "model.pt")), split=split, f2=bool(a.f2), lane_gt=os.path.basename(LT.LD.lane_store()[1]), table_version=LT.TABLE_VERSION, node_stage=("mnms" if m.mnms else "plain")))
                fresh = False
            arrays = {n_: np.load(_path(out, n_), mmap_mode="r+") for n_ in FIELDS if os.path.exists(_path(out, n_))}
        B = tk["cell"].shape[0]
        for j in range(B):
            lab = label_frame(pos_all[j], valid_all[j], b["table"][j], a.r, K=a.K)
            for n_, v in lab.items():
                arrays[n_][i + j] = v
            for n_, v in tk.items():
                arrays[n_][i + j] = v[j] if n_ in ("n_e50", "n_e20") else v[j, :a.K]
        i += B
        nb += 1
        if nb % 50 == 0 or i >= n_all:
            for ar in arrays.values():
                ar.flush()
            _atomic_json(prog, dict(done=i, total=n_all, seconds=round(time.time() - t0, 1)))
            print(f"   {split}: {i}/{n_all} frames, {time.time() - t0:.0f} s", flush=True)
    assert i == n_all, (i, n_all)
    _atomic_json(os.path.join(out, "manifest.json"), dict(json.load(open(os.path.join(out, "meta.json"))), frames=n_all, seconds=round(time.time() - t0, 1), finished=time.strftime("%Y-%m-%d %H:%M:%S")))
    print(f"cache {out}: {n_all} frames")
    return out


def _collate(b):
    return dict(x=torch.stack([o["x"] for o in b]), row=[o["row"] for o in b], token=[o["token"] for o in b], table=[{k: o[k] for k in ("pos", "dirv", "cell", "flags", "piece", "sstart", "src", "dst", "long", "scale", "conn")} for o in b])


# ----------------------------------------------------------------------------------------------------------- report
def report(d, ks=KS):
    c = TokCache(d)
    n = c.done or len(c)
    ng, nm = c.f["n_gt"][:n].astype(float), c.f["n_gt_matched"][:n].astype(float)
    nl, lost = c.f["n_links"][:n].astype(float), c.f["n_lost"][:n].astype(float)
    k_ = c.K
    out = dict(frames=n, K=k_, gt_nodes=int(ng.sum()), matched=round(float(nm.sum() / max(ng.sum(), 1)), 4), links_kept=round(float(1 - lost.sum() / max(nl.sum(), 1)), 4),
               deliv_cut=int(((c.f["n_e50"][:n] > k_)).sum()), deliv_cut_nodes=int(np.maximum(c.f["n_e50"][:n].astype(int) - k_, 0).sum()), e50_mean=round(float(c.f["n_e50"][:n].mean()), 1), e20_max=int(c.f["n_e20"][:n].max()))
    out["at_K"] = {int(k): dict(matched=round(float(c.f["match_at"][:n, j].sum() / max(ng.sum(), 1)), 4), links_kept=round(float(1 - c.f["lost_at"][:n, j].sum() / max(nl.sum(), 1)), 4)) for j, k in enumerate(ks)}
    sel = (c.f["e_p"][:n] >= 0.5) & c.f["valid"][:n]
    gt = c.f["gt_id"][:n] >= 0
    out["delivered_tokens"] = int(sel.sum())
    out["precision_at_r"] = round(float((sel & gt).sum() / max(sel.sum(), 1)), 4)
    return out


# ----------------------------------------------------------------------------------------------------------- selftest
def selftest():
    torch.manual_seed(0)
    B, C = 2, 4
    e = torch.randn(B, 50, 50)
    e[0, 3, 3] = e[0, 0, 5] = 9.0                                                                           # a tie: the lower cell first
    no = dict(e=e, e_raw=e + 1, pos=torch.randn(B, 2, 50, 50), off=torch.randn(B, 2, 50, 50), dir_raw=torch.randn(B, 2, 50, 50), flags=torch.randn(B, 8, 50, 50), s=torch.randn(B, 50, 50), uout=torch.randn(B, 50, 50),
              uin=torch.randn(B, 50, 50), tok=torch.randn(B, C, 50, 50))
    f2 = torch.randn(B, 6, 50, 50)
    t = select_tokens(no, f2, 7)
    assert t["cell"].dtype == torch.int16 and t["tok"].dtype == torch.float16 and t["f2"].shape == (B, 7, 6) and t["pos"].shape == (B, 7, 2)
    fp, sp, ep_ = no["flags"].sigmoid(), no["s"].sigmoid(), e.sigmoid()
    for b in range(B):                                                                                       # against a plain loop
        flat = e[b].flatten()
        want = sorted(range(2500), key=lambda c: (-float(flat[c]), c))[:7]
        assert t["cell"][b].tolist() == want, (t["cell"][b].tolist(), want)
        for k, c in enumerate(want):
            i, j = divmod(c, 50)
            assert torch.equal(t["e"][b, k], e[b, i, j]) and torch.equal(t["e_raw"][b, k], (e + 1)[b, i, j]) and torch.equal(t["pos"][b, k], no["pos"][b, :, i, j])
            assert torch.equal(t["flags_p"][b, k], fp[b, :, i, j]) and torch.equal(t["s_p"][b, k], sp[b, i, j]) and torch.equal(t["e_p"][b, k], ep_[b, i, j])             # the whole-map sigmoid, as predict makes it (a scalar sigmoid can differ by an ulp)
            assert torch.equal(t["tok"][b, k], no["tok"][b, :, i, j].to(torch.float16)) and torch.equal(t["f2"][b, k], f2[b, :, i, j].to(torch.float16))
            v = no["dir_raw"][b, :, i, j]
            assert torch.allclose(t["dirv"][b, k], v / v.norm(), atol=1e-6) and abs(float(t["dirv"][b, k].norm()) - 1) < 1e-5
            assert bool(t["valid"][b, k]) == (float(e[b, i, j].sigmoid()) > P_ACTIVE)
    assert t["cell"][0, 0] == 5 and t["cell"][0, 1] == 3 * 50 + 3, "tie order: the lower flat cell first"
    assert t["n_e50"].tolist() == [int((e[b].sigmoid() >= 0.5).sum()) for b in range(B)]
    assert select_tokens(no, None, 3).get("f2") is None
    # assignment: unique, closest first, radius inclusive, invalid tokens never match
    g, d = assign_unique(np.array([[0, 0], [0.5, 0], [10, 0]]), np.array([True, True, True]), np.array([[0.4, 0], [10.2, 0]]), 1.5)
    assert g.tolist() == [-1, 0, 1] and abs(d[1] - 0.1) < 1e-12 and np.isnan(d[0]), (g, d)               # token 1 is closer than token 0: it wins, token 0 stays without
    g, _ = assign_unique(np.array([[0, 0], [0.5, 0]]), np.array([False, True]), np.array([[0.1, 0]]), 1.5)
    assert g.tolist() == [-1, 0]
    g, _ = assign_unique(np.array([[0, 0]]), np.array([True]), np.array([[1.5, 0]]), 1.5)
    assert g.tolist() == [0]                                                                                # exactly r is inside
    g, _ = assign_unique(np.array([[0, 0]]), np.array([True]), np.array([[1.5001, 0]]), 1.5)
    assert g.tolist() == [-1]
    g, _ = assign_unique(np.array([[1.0, 0], [-1.0, 0]]), np.array([True, True]), np.array([[0.0, 0]]), 1.5)
    assert g.tolist() == [0, -1], "tie in distance: the lower token wins"
    assert assign_unique(np.zeros((0, 2)), np.zeros(0, bool), np.zeros((3, 2)), 1.5)[0].shape == (0,)
    # types
    fl = np.zeros((6, 8), bool)
    fl[1, 2] = True                                                                                         # split -> fork
    fl[2, 3] = True                                                                                         # merge
    fl[3, 0] = True                                                                                         # stop -> entry
    fl[4, 1] = True                                                                                         # exit
    fl[5, 6] = fl[5, 2] = True                                                                              # win_in + split -> fork wins
    assert types_of(fl).tolist() == [1, 2, 3, 4, 5, 2]
    fl2 = np.zeros((2, 8), bool)
    fl2[0, 4] = fl2[1, 7] = True
    assert types_of(fl2).tolist() == [6, 6]
    # labels + links on tokens: a chain of 3 GT nodes (the last link long), tokens on nodes 0 and 2 only
    table = dict(pos=np.array([[0.0, 0], [2, 0], [20, 0]]), flags=np.zeros((3, 8), bool), sstart=np.array([True, False, False]), src=np.array([0, 1]), dst=np.array([1, 2]), long=np.array([False, True]),
                 conn=np.array([False, True]), scale=np.array([[np.nan, np.nan], [0.3, 0.4]]))
    table["flags"][2, 0] = True
    tpos = np.array([[0.2, 0], [20.1, 0], [50, 50], [2.1, 0]])
    lab = label_frame(tpos, np.array([True, True, True, False]), table, 1.5, ks=(1, 2, 4))
    assert lab["gt_id"].tolist() == [0, 2, -1, -1] and lab["gt_type"].tolist() == [1, 4, 0, 0] and lab["n_gt"] == 3 and lab["n_gt_matched"] == 2 and lab["n_links"] == 2 and lab["n_lost"] == 2
    assert lab["match_at"].tolist() == [1, 2, 2] and lab["lost_at"].tolist() == [2, 2, 2] and lab["gt_aux"][0].tolist() == [True, False, False] and lab["gt_aux"][1].tolist() == [False, False, True]
    assert lab["link_tok"][0].tolist() == [-1, -1]
    tpos2 = np.array([[0.2, 0], [20.1, 0], [50, 50], [2.1, 0]])
    lab = label_frame(tpos2, np.array([True, True, True, True]), table, 1.5, ks=(2, 4))                      # now the middle node has token 3: both links are on tokens
    assert lab["gt_id"].tolist() == [0, 2, -1, 1] and lab["n_lost"] == 0 and lab["link_tok"][:2].tolist() == [[0, 3], [3, 1]] and lab["link_long"][:2].tolist() == [False, True] and lab["link_conn"][:2].tolist() == [False, True]
    assert np.allclose(lab["link_scale"][1], [0.3, 0.4]) and np.isnan(lab["link_scale"][0]).all() and lab["lost_at"].tolist() == [2, 0] and lab["match_at"].tolist() == [2, 3]
    empty = dict(pos=np.zeros((0, 2)), flags=np.zeros((0, 8), bool), sstart=np.zeros(0, bool), src=np.zeros(0, int), dst=np.zeros(0, int), long=np.zeros(0, bool), conn=np.zeros(0, bool), scale=np.zeros((0, 2)))
    lab = label_frame(tpos, np.ones(4, bool), empty, 1.5, ks=(2,))
    assert (lab["gt_id"] == -1).all() and lab["n_gt"] == 0 and lab["n_links"] == 0
    # the dump comparison on a toy cache: equal passes, any difference fails
    class Toy:
        K, rows = 4, [7]
        f = dict(cell=np.array([[53, 5, 100, 9]], np.int16), e_p=np.array([[0.9, 0.6, 0.3, 0.01]], np.float32), pos=np.array([[[3.0, 5.0], [-48.0, -39.0], [1.0, 1.0], [0.0, 0.0]]], np.float32),
                 dirv=np.array([[[1.0, 0], [0, 1.0], [1.0, 0], [1.0, 0]]], np.float32), flags_p=np.full((1, 4, 8), 0.25, np.float32), s_p=np.full((1, 4), 0.5, np.float32))
    toy = Toy()
    toy.f["pos"][0, 0] = [-45.0, -43.0]                                                                      # cell (2, 3)? recomputed below to match cell 53 = (1, 3)
    toy.f["pos"][0, 0] = [-47.0, -43.0]                                                                      # row (x+50)/2 = 1.5 -> 1, col (y+50)/2 = 3.5 -> 3: cell 53
    toy.f["pos"][0, 1] = [-49.5, -39.0]                                                                      # row 0, col 5: cell 5
    toy.f["pos"][0, 2] = [-1.0, -1.0]                                                                        # row 24, col 24 -> 24 * 50 + 24 = 1224 (not 100: the toy cache is wrong on purpose below)
    toy.f["cell"][0, 2] = 1224

    def dump_of(t):
        return [dict(row=7, e=t.f["e_p"][0, :3].astype(np.float64), pos=t.f["pos"][0, :3].astype(np.float64), dirv=t.f["dirv"][0, :3].astype(np.float64), flags=t.f["flags_p"][0, :3].astype(np.float64),
                     s=t.f["s_p"][0, :3].astype(np.float64))]
    ok, rep = compare_dump(toy, dump_of(toy))
    assert ok and rep["nodes_compared"] == 3 and rep["worst"]["e"] == 0.0, rep
    for key, mut in (("e", lambda dd: dd[0]["e"].__setitem__(1, dd[0]["e"][1] + 1e-12)), ("pos", lambda dd: dd[0]["pos"].__setitem__((0, 0), dd[0]["pos"][0, 0] + 1e-9)), ("flags", lambda dd: dd[0]["flags"].__setitem__((2, 3), 0.2500001)),
                     ("s", lambda dd: dd[0]["s"].__setitem__(0, 0.5000001)), ("dirv", lambda dd: dd[0]["dirv"].__setitem__((1, 1), 0.9999999))):
        dd = dump_of(toy)
        mut(dd)
        ok, rep = compare_dump(toy, dd)
        assert not ok and rep["mismatches"] > 0 or rep["worst"][key] > 0, (key, rep)
    dd = dump_of(toy)
    dd[0]["row"] = 8
    assert not compare_dump(toy, dd)[0]
    dd = dump_of(toy)
    dd[0]["pos"][1] = [20.0, 20.0]                                                                           # another cell
    assert not compare_dump(toy, dd)[0]
    # the qformat nodes of a frame
    fr = dict(valid=np.array([True, False, True]), cell=np.array([53, 5, 1224], np.int16), pos=np.zeros((3, 2), np.float32), dirv=np.ones((3, 2), np.float32), e_p=np.array([0.9, 0.1, 0.6], np.float32),
              flags_p=np.zeros((3, 8), np.float32), s_p=np.zeros(3, np.float32))
    nd = nodes_of(fr)
    assert nd["cell"].tolist() == [[1, 3], [24, 24]] and nd["e"].tolist() == [np.float32(0.9).astype(float), np.float32(0.6).astype(float)] and nd["section"].tolist() == [0, 1]
    print("lane_tokcache selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("build")
    s.add_argument("--run", required=True)
    s.add_argument("--split", required=True, choices=("val", "val_clean", "train"))
    s.add_argument("--K", type=int, default=320)
    s.add_argument("--no-f2", dest="f2", action="store_false")
    s.add_argument("--r", type=float, default=1.5)
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--workers", type=int, default=2)
    s = sub.add_parser("fixture")
    s.add_argument("--cache", required=True)
    s.add_argument("--dump", required=True)
    s.add_argument("--prefix", action="store_true", help="the cache is a limited build: compare the first frames of the dump only")
    s = sub.add_parser("report")
    s.add_argument("--cache", required=True)
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "build":
        build(a)
    elif a.cmd == "report":
        print(json.dumps(report(a.cache), indent=1))
    else:
        c = TokCache(a.cache)
        ok, rep = compare_dump(c, pickle.load(open(a.dump if os.path.isabs(a.dump) else os.path.join(ROOT, a.dump), "rb")), prefix=a.prefix)
        print(("FIXTURE PASS " if ok else "FIXTURE FAIL ") + json.dumps(rep))
        sys.exit(0 if ok else 1)
