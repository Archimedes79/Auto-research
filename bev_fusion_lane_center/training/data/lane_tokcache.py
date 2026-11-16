"""D113 / E365 predicted-node cache: the tokens of a FROZEN lane head B (a `QGraph` run: B' = `runs/e292_a_s<S>` with the Matrix-NMS layer, plain B = `runs/e273_p_s<S>`) on every frame of TRAIN and VAL, with the GT labels
of every token (spec: section 2a of `experiments/E365_e2e-token-champion-s1/experiment.md`). Derived data, not in git; this tool, its selftest and the manifests are.

    wsl bash tools/env/_pyr.sh lane_tokcache.py build   --run runs/e292_a_s0 --split val|train [--K 384] [--no-f2] [--limit N]   # GPU, resumable; -> runs/e365_cache/b<id>_<S>_K<K>_<GT signature>_<version>/<split>/
    wsl bash tools/env/_py14.sh lane_tokcache.py fixture --cache <dir> --dump temp/e292/predfull_e292_a_s0.pkl [--prefix]         # the cache against B's stored dump, bit for bit (VAL)
    wsl bash tools/env/_py14.sh lane_tokcache.py report  --cache <dir> [--write]                                                  # the manifest numbers (--write: manifest.json)
    wsl bash tools/env/_py14.sh lane_tokcache.py selftest
    c = TokCache(dir); f = c[i]                                                                                                   # one frame as numpy arrays; `lane_ptok_data.py` is the loader

Frames: the order of `lane_tok_data.TokData(split, res=2.0)` (TRAIN = the rows of `QData('train')` with a stored lane GT, VAL = exactly the 873 rows of data/splits.json['val'], asserted there); the GT table of a frame is the E357 table
cache on the 2 m lattice (B's own cells). Tokens of a frame: the K cells with the largest existence logit `e` of `QGraph.node_stage` (B': after the Matrix-NMS layer; `e_raw` = the head's own logit), a stable sort, equal logits ->
the lower flat cell; `valid` = sigmoid(e) > 0.05 (B's own active threshold), the other rows are padding. Per token (float32 unless said): cell (int16, flat row * 50 + col, row = x, col = y), e, e_raw, e_p = sigmoid(e) as the GPU
makes it from the whole map, pos (2, metres = cell centre + off), off (2), dirv (2, unit of dir_raw), dir_raw (2), flags (8 logits, order `qtargets.FLAGS`), flags_p (8 probabilities), s, s_p, uout, uin, tok (float16, 64: the refined
node token that B's link head reads), f2 (float16, 144: the 2 m pyramid feature of the cell; optional). B is frozen: nothing here carries a gradient.
Labels (r = 1.5 m, cosine of the headings >= 0.5; a valid token is NEAR a GT node when both hold): gt_id (int32, the row of the frame's GT table, -1 none) and gd (metres, NaN none): the tokens in descending existence take, one after
the other, the NEAREST still unassigned GT node near them (ties: the lower GT node) = a function of the input; gt_type (int8, `TYPES`: 0 none, 1 lane point, 2 fork (split), 3 merge, 4 entry (stop), 5 exit, 6 boundary (start, end, win_in,
win_out); several flags: the first of split > merge > stop > exit > boundary), gt_flags (8 bool: the raw flags of the assigned node), gt_aux (3 bool: section start, start of a long link, end of a long link). A GT node is COVERED when
a valid token is near it. Link targets: the GT links are contracted over the uncovered GT nodes (`jtok_eval.sub_table`: a chain through removed nodes becomes one link, class long = chord > 8 m, fitted scale only for direct links); for every
contracted link a -> b with a token i assigned to a: the pairs (i, j) for every valid token j near b (assigned or not; j != i): stored sparse as ypair (token i, token j; -1 pad), ypair_long, ypair_scale (NaN unless a fitted scale
exists), ypair_gate (the pair is inside the allowed gate of `jtok.pair_geometry` on the stored tokens); `lane_ptok_data` makes the dense y / ylong / yscale of the spec from them (a pair that two links claim with two classes is kept as
long; counted in n_conflict). Per frame counts: n_gt, n_gt_assigned, n_links (GT links), n_links_c (contracted), gt_out (GT links with an end covered by no token), n_nosrc (contracted links without a source token), n_reach (contracted links
with one pair inside the gate), n_pairs, n_conflict, n_valid, n_deliv (valid tokens with sigmoid(e) >= 0.5), n_deliv_near (of them near a GT node), n_gt_deliv_cov (GT nodes with a delivered token near), n_e50 / n_e20 (cells of the
whole map with sigmoid(e) >= 0.5 / 0.2), n_e50_in (those of them inside the K tokens), cov_nodes_at / cov_links_at (F, 4): GT nodes covered and GT links with both ends covered when only the first KST[j] tokens are kept.
Directory: one .npy per field (memory maps), frames.json, meta.json, progress.json (complete batches of 4: a build resumes), manifest.json. Version `v1`: bump VERSION by hand when the table, the assignment or the label code changes.
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
CACHE = os.path.join(ROOT, "runs", "e365_cache")
TYPES = ("none", "lane", "fork", "merge", "entry", "exit", "boundary")
FLAGS = ("stop", "exit", "split", "merge", "start", "end", "win_in", "win_out")                  # = qtargets.FLAGS (asserted in `build`)
KST = (256, 320, 384, 512)                                                                       # the token budgets of the coverage statistics
N, RES, HALF = 50, 2.0, 50.0
P_ACTIVE = 0.05                                                                                    # B's own active threshold (qlinks.LinkHead: valid = sigmoid(e) > thr)
R_NEAR, COS_NEAR = 1.5, 0.5
PMAX = 4096                                                                                        # most stored target pairs of a frame (asserted)


# ----------------------------------------------------------------------------------------------------------- pure parts (selftest)
def select_tokens(no, f2_map, K):
    """node-stage outputs `no` (the dict of `QGraph.node_stage`, a batch) and the 2 m pyramid map `f2_map` (B, C, 50, 50) or None -> dict of tensors (B, K, ...) of the fields in the module docstring; the probabilities are made on the
    whole maps first and gathered afterwards, exactly as `qpredict.predict_from_out` does, so that they are the same numbers"""
    e = no["e"].float().flatten(1)
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


def near_matrix(tpos, tdirv, tvalid, gpos, gdirv, r=R_NEAR, cos=COS_NEAR):
    """valid tokens (T, 2) / (T, 2) / (T,) and GT nodes (M, 2) / (M, 2) -> (near (T, M) bool, d (T, M)): within r metres (inclusive) and the cosine of the headings at least `cos`"""
    tpos, gpos = np.asarray(tpos, float), np.asarray(gpos, float)
    d = np.linalg.norm(tpos[:, None, :] - gpos[None, :, :], axis=2)
    cs = np.asarray(tdirv, float) @ np.asarray(gdirv, float).T
    return np.asarray(tvalid, bool)[:, None] & (d <= r) & (cs >= cos), d


def assign_greedy(near, d, K):
    """the first K tokens in the given (descending existence) order take, one after the other, the NEAREST still unassigned GT node near them (ties: the lower GT node) -> (gt_id (K,) int, gd (K,) float, NaN none)"""
    M = near.shape[1]
    gt, gd, used = -np.ones(K, int), np.full(K, np.nan), np.zeros(M, bool)
    for t in range(min(K, near.shape[0])):
        cand = near[t] & ~used
        if cand.any():
            g = int(np.argmin(np.where(cand, d[t], np.inf)))
            gt[t], gd[t], used[g] = g, d[t, g], True
    return gt, gd


def types_of(flags):
    """GT flags (n, 8) bool in the order of `FLAGS` -> (n,) int8 of `TYPES` (1 lane point when no flag): split -> fork, merge, stop -> entry, exit, start / end / win_in / win_out -> boundary; priority split > merge > stop > exit > boundary"""
    f = np.asarray(flags, bool).reshape(-1, 8)
    t = np.ones(len(f), np.int8)
    for code, cols in ((6, (4, 5, 6, 7)), (5, (1,)), (4, (0,)), (3, (3,)), (2, (2,))):                       # lowest priority first, later rows overwrite
        t[f[:, list(cols)].any(1)] = code
    return t


def label_frame(tk, table, r=R_NEAR, cos=COS_NEAR, K=None, allowed=None, ks=KST):
    """the labels of one frame. tk: numpy dict of the tokens in rank order (pos, dirv, valid, e_p; T >= K rows), table: the GT table of `lane_tok_data` (pos, dirv, flags, sstart, src, dst, long, conn, scale, cell, piece),
    allowed: (K, K) bool, the gate of `jtok.pair_geometry` on the first K tokens (None: all False) -> dict of the label fields (module docstring); the coverage statistics use the first ks[j] of the T given tokens"""
    from jtok_eval import sub_table
    T = len(tk["pos"])
    K = T if K is None else K
    M = len(table["pos"])
    near, d = near_matrix(tk["pos"], tk["dirv"], tk["valid"], table["pos"], table["dirv"], r, cos)
    gt, gd = assign_greedy(near, d, K)
    got = gt >= 0
    src, dst = np.asarray(table["src"], int), np.asarray(table["dst"], int)
    long_ = np.asarray(table["long"], bool)
    uout, uin = np.zeros(M, bool), np.zeros(M, bool)
    uout[src[long_]], uin[dst[long_]] = True, True
    gflags, gaux = np.zeros((K, 8), bool), np.zeros((K, 3), bool)
    if got.any():
        gflags[got] = np.asarray(table["flags"], bool)[gt[got]]
        gaux[got] = np.stack([np.asarray(table["sstart"], bool)[gt[got]], uout[gt[got]], uin[gt[got]]], 1)
    gtype = np.zeros(K, np.int8)
    gtype[got] = types_of(gflags[got])
    covered = near[:K].any(0)
    sub = sub_table(table, covered)
    kept = np.flatnonzero(covered)
    pair, plong, pscale, pgate, conflict, nosrc, reach = {}, {}, {}, {}, 0, 0, 0
    tok_of = -np.ones(M, int)
    tok_of[gt[got]] = np.flatnonzero(got)
    for l in range(len(sub["src"])):
        a, b = int(kept[sub["src"][l]]), int(kept[sub["dst"][l]])
        i = int(tok_of[a])
        if i < 0:
            nosrc += 1
            continue
        lg, sc, ok = bool(sub["long"][l]), np.asarray(sub["scale"][l], np.float32), False
        for j in np.flatnonzero(near[:K, b]):
            if j == i:
                continue
            g_ = bool(allowed[i, j]) if allowed is not None else False
            ok = ok or g_
            if (i, int(j)) in plong:
                conflict += int(plong[(i, int(j))] != lg)
                if plong[(i, int(j))] or not lg:
                    continue
            plong[(i, int(j))], pscale[(i, int(j))], pgate[(i, int(j))] = lg, sc, g_
        reach += int(ok)
    n = len(plong)
    assert n <= PMAX, f"{n} target pairs exceed PMAX {PMAX}"
    yp = -np.ones((PMAX, 2), np.int16)
    yl, yg = np.zeros(PMAX, bool), np.zeros(PMAX, bool)
    ys = np.full((PMAX, 2), np.nan, np.float32)
    for k, key in enumerate(sorted(plong)):
        yp[k], yl[k], yg[k], ys[k] = key, plong[key], pgate[key], pscale[key]
    both = lambda cov: int((cov[src] & cov[dst]).sum())  # noqa: E731
    valid_k, ep_k = np.asarray(tk["valid"][:K], bool), np.asarray(tk["e_p"][:K])
    deliv = valid_k & (ep_k >= 0.5)
    return dict(gt_id=gt.astype(np.int32), gd=gd.astype(np.float32), gt_type=gtype, gt_flags=gflags, gt_aux=gaux, ypair=yp, ypair_long=yl, ypair_scale=ys, ypair_gate=yg,
                n_gt=np.int32(M), n_gt_assigned=np.int32(got.sum()), n_links=np.int32(len(src)), n_links_c=np.int32(len(sub["src"])), gt_out=np.int32(len(src) - both(covered)), n_nosrc=np.int32(nosrc),
                n_reach=np.int32(reach), n_pairs=np.int32(n), n_conflict=np.int32(conflict), n_valid=np.int32(valid_k.sum()), n_deliv=np.int32(deliv.sum()),
                n_deliv_near=np.int32(near[:K][deliv].any(1).sum()), n_gt_deliv_cov=np.int32(near[:K][deliv].any(0).sum()), n_e50_in=np.int32((deliv).sum()),
                cov_nodes_at=np.array([near[:min(k, T)].any(0).sum() for k in ks], np.int32), cov_links_at=np.array([both(near[:min(k, T)].any(0)) for k in ks], np.int32))


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


def nodes_of(frame, sel=None):
    """the qformat 'nodes' dict (`qformat.py`) of one cached frame, the tokens of `sel` (default: the valid ones): pos, dirv, e (probability), flags (probabilities), s (probability), cell (row, col), section = every node 0
    (the section-free read-out of E357)"""
    sel = np.nonzero(frame["valid"])[0] if sel is None else np.asarray(sel)
    c = frame["cell"][sel].astype(int)
    return dict(pos=frame["pos"][sel].astype(float), dirv=frame["dirv"][sel].astype(float), e=frame["e_p"][sel].astype(float), flags=frame["flags_p"][sel].astype(float), s=frame["s_p"][sel].astype(float),
                cell=np.stack([c // N, c % N], 1), section=np.zeros(len(sel), int))


# ----------------------------------------------------------------------------------------------------------- storage
FIELDS = dict(cell="int16", valid="bool", e="float32", e_raw="float32", e_p="float32", pos="float32", off="float32", dirv="float32", dir_raw="float32", flags="float32", flags_p="float32", s="float32", s_p="float32",
              uout="float32", uin="float32", tok="float16", f2="float16", gt_id="int32", gd="float32", gt_type="int8", gt_flags="bool", gt_aux="bool", ypair="int16", ypair_long="bool", ypair_scale="float32",
              ypair_gate="bool", n_gt="int32", n_gt_assigned="int32", n_links="int32", n_links_c="int32", gt_out="int32", n_nosrc="int32", n_reach="int32", n_pairs="int32", n_conflict="int32", n_valid="int32",
              n_deliv="int32", n_deliv_near="int32", n_gt_deliv_cov="int32", n_e50_in="int32", n_e50="int32", n_e20="int32", cov_nodes_at="int32", cov_links_at="int32")


def _abs(p):
    """a path relative to the project root (the wrappers run a script from its own folder)"""
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def _path(d, name):
    return os.path.join(d, name + ".npy")


def _atomic_json(path, obj):
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


class TokCache:
    """read access to a cache directory: `c.rows`, `c.tokens`, `c.K`, `c.f` (the memory maps by field name), `c[i]` one frame as a dict of numpy arrays (copies), `c.done` frames written"""

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


def cache_name(run_name, K, sig):
    """`runs/e365_cache/b<id>_<S>_K<K>_<GT signature>_<version>` of a run folder name like e292_a_s0 (id e292_a, S 0)"""
    ident, _, seed = run_name.rpartition("_s")
    assert ident and seed.isdigit(), f"run name {run_name!r} is not <id>_s<seed>"
    return f"b{ident}_{seed}_K{K}_{sig}_{VERSION}"


def _shapes(K, d_tok, d_f2):
    s = dict(cell=(K,), valid=(K,), e=(K,), e_raw=(K,), e_p=(K,), pos=(K, 2), off=(K, 2), dirv=(K, 2), dir_raw=(K, 2), flags=(K, 8), flags_p=(K, 8), s=(K,), s_p=(K,), uout=(K,), uin=(K,), tok=(K, d_tok),
             gt_id=(K,), gd=(K,), gt_type=(K,), gt_flags=(K, 8), gt_aux=(K, 3), ypair=(PMAX, 2), ypair_long=(PMAX,), ypair_scale=(PMAX, 2), ypair_gate=(PMAX,), cov_nodes_at=(len(KST),), cov_links_at=(len(KST),))
    for n in ("n_gt", "n_gt_assigned", "n_links", "n_links_c", "gt_out", "n_nosrc", "n_reach", "n_pairs", "n_conflict", "n_valid", "n_deliv", "n_deliv_near", "n_gt_deliv_cov", "n_e50_in", "n_e50", "n_e20"):
        s[n] = ()
    if d_f2:
        s["f2"] = (K, d_f2)
    return s


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()[:16]


def _collate(b):
    return dict(x=torch.stack([o["x"] for o in b]), row=[o["row"] for o in b], token=[o["token"] for o in b], table=[{k: o[k] for k in ("pos", "dirv", "cell", "flags", "piece", "sstart", "src", "dst", "long", "scale", "conn")} for o in b])


def build(a):
    import jtok as JT
    import lane_tok_data as LT
    import qpredict as QP
    import qtargets as QT
    assert tuple(QT.FLAGS) == FLAGS
    split = {"val": "val", "val_clean": "val", "train": "train"}[a.split]
    run = _abs(a.run)
    sig = os.path.basename(LT.LD.lane_store()[1])
    out = os.path.join(CACHE, cache_name(os.path.basename(os.path.normpath(run)), a.K, sig) + (f"_first{a.limit}" if a.limit else ""), split)        # a limited build never shares a directory with the full one
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = LT.TokData(split, res=2.0, feats=True)
    n_all = len(ds) if not a.limit else min(a.limit, len(ds))
    m = QP.load_model(run, dev)
    bs, kfull = 4, max(a.K, max(KST))
    prog = os.path.join(out, "progress.json")
    done = json.load(open(prog))["done"] if os.path.exists(prog) else 0
    fresh = not os.path.exists(os.path.join(out, "meta.json"))
    assert done % bs == 0 and done <= n_all, f"progress {done} is not a multiple of {bs} within {n_all}"
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, list(range(done, n_all))), batch_size=bs, shuffle=False, num_workers=a.workers, collate_fn=_collate,
                                     persistent_workers=a.workers > 0, prefetch_factor=2 if a.workers > 0 else None)
    arrays, t0, nb, i = None, time.time(), 0, done
    os.makedirs(out, exist_ok=True)
    _atomic_json(os.path.join(out, "frames.json"), dict(rows=ds.rows[:n_all], tokens=ds.tokens[:n_all]))
    for b in dl:
        with torch.no_grad():
            x = b["x"].to(dev).float()
            P = m.pyr(x)
            no = m.node_stage(P)
            tkt = select_tokens(no, P["feat2"] if a.f2 else None, kfull)                                       # all the tokens the coverage statistics need, K of them are stored
            allowed = JT.pair_geometry(tkt["pos"][:, :a.K], tkt["dirv"][:, :a.K], tkt["valid"][:, :a.K])[1].cpu().numpy()
        tk = {k: v.cpu().numpy() for k, v in tkt.items()}
        if arrays is None:
            if fresh:
                d_tok, d_f2 = tk["tok"].shape[-1], (tk["f2"].shape[-1] if "f2" in tk else 0)
                os.makedirs(out, exist_ok=True)
                for name_, shp in _shapes(a.K, d_tok, d_f2).items():
                    np.lib.format.open_memmap(_path(out, name_), mode="w+", dtype=FIELDS[name_], shape=(n_all,) + shp).flush()
                _atomic_json(os.path.join(out, "meta.json"), dict(run=a.run, run_sha=_sha(os.path.join(run, "model.pt")), split=split, K=a.K, f2=bool(a.f2), r=R_NEAR, cos=COS_NEAR, version=VERSION, lane_gt=sig,
                                                                  table_version=LT.TABLE_VERSION, node_stage="mnms" if m.mnms else "plain", kst=list(KST), types=list(TYPES),
                                                                  type_priority="split > merge > stop > exit > boundary (start, end, win_in, win_out) > lane point"))
                fresh = False
            arrays = {n_: np.load(_path(out, n_), mmap_mode="r+") for n_ in FIELDS if os.path.exists(_path(out, n_))}
        B = tk["cell"].shape[0]
        for j in range(B):
            lab = label_frame({k: v[j] for k, v in tk.items() if k in ("pos", "dirv", "valid", "e_p")}, b["table"][j], K=a.K, allowed=allowed[j])
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
    man = dict(json.load(open(os.path.join(out, "meta.json"))), frames=n_all, seconds=round(time.time() - t0, 1), finished=time.strftime("%Y-%m-%d %H:%M:%S"), numbers=report(out))
    _atomic_json(os.path.join(out, "manifest.json"), man)
    print(f"cache {out}: {n_all} frames")
    return out


# ----------------------------------------------------------------------------------------------------------- report
def report(d):
    """the manifest numbers of a finished cache directory (spec 2a): GT nodes per frame, the share of GT nodes covered and of GT links with both ends covered at every K of `KST`, the share of GT nodes assigned and of valid tokens
    assigned, B''s node recall and precision at r (delivered tokens = valid, sigmoid(e) >= 0.5, inside K), the share of the cells with sigmoid(e) >= 0.5 outside K, the contracted links with a source token and inside the gate (the
    recall ceiling of the links), the K rule (K = 384 unless the coverage at 384 is below 0.99 x that at 512 on nodes or links)"""
    c = TokCache(d)
    n = c.done or len(c)
    s = lambda name: c.f[name][:n].astype(np.int64)  # noqa: E731
    ng, nl, nlc = s("n_gt").sum(), s("n_links").sum(), s("n_links_c").sum()
    cn, cl = c.f["cov_nodes_at"][:n].sum(0).astype(float), c.f["cov_links_at"][:n].sum(0).astype(float)
    cov = {int(k): dict(nodes=round(float(cn[j] / max(ng, 1)), 4), links_both_ends=round(float(cl[j] / max(nl, 1)), 4)) for j, k in enumerate(KST)}
    nm = c.f["gt_id"][:n] >= 0
    ratio = min(cov[384]["nodes"] / max(cov[512]["nodes"], 1e-9), cov[384]["links_both_ends"] / max(cov[512]["links_both_ends"], 1e-9))
    return dict(frames=n, K=c.K, gt_nodes_per_frame=round(float(ng / n), 1), gt_links_per_frame=round(float(nl / n), 1), valid_tokens_per_frame=round(float(s("n_valid").mean()), 1), cells_e50_per_frame=round(float(s("n_e50").mean()), 1),
                coverage_at_K=cov, k_rule_ratio_384_over_512=round(float(ratio), 4), k_rule_choice=384 if ratio >= 0.99 else 512,
                gt_nodes_assigned=round(float(s("n_gt_assigned").sum() / max(ng, 1)), 4), tokens_assigned_of_valid=round(float(nm.sum() / max(s("n_valid").sum(), 1)), 4),
                b_node_recall_at_r=round(float(s("n_gt_deliv_cov").sum() / max(ng, 1)), 4), b_node_precision_at_r=round(float(s("n_deliv_near").sum() / max(s("n_deliv").sum(), 1)), 4),
                e50_cells_outside_K=round(float(1 - s("n_e50_in").sum() / max(s("n_e50").sum(), 1)), 4), frames_with_e50_outside_K=int((s("n_e50") > s("n_e50_in")).sum()),
                links_contracted_per_frame=round(float(nlc / n), 1), links_without_source_token=round(float(s("n_nosrc").sum() / max(nlc, 1)), 4), links_reachable_in_gate=round(float(s("n_reach").sum() / max(nlc, 1)), 4),
                links_reachable_of_gt_links=round(float(s("n_reach").sum() / max(nl, 1)), 4), gt_links_lost_end_uncovered=round(float(s("gt_out").sum() / max(nl, 1)), 4), pairs_per_frame=round(float(s("n_pairs").mean()), 1),
                pairs_max=int(s("n_pairs").max()), pair_class_conflicts=int(s("n_conflict").sum()))


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
            assert torch.equal(t["flags_p"][b, k], fp[b, :, i, j]) and torch.equal(t["s_p"][b, k], sp[b, i, j]) and torch.equal(t["e_p"][b, k], ep_[b, i, j])             # the whole-map sigmoid, as predict makes it
            assert torch.equal(t["tok"][b, k], no["tok"][b, :, i, j].to(torch.float16)) and torch.equal(t["f2"][b, k], f2[b, :, i, j].to(torch.float16))
            v = no["dir_raw"][b, :, i, j]
            assert torch.allclose(t["dirv"][b, k], v / v.norm(), atol=1e-6) and abs(float(t["dirv"][b, k].norm()) - 1) < 1e-5
            assert bool(t["valid"][b, k]) == (float(e[b, i, j].sigmoid()) > P_ACTIVE)
    assert t["cell"][0, 0] == 5 and t["cell"][0, 1] == 3 * 50 + 3, "tie order: the lower flat cell first"
    assert t["n_e50"].tolist() == [int((e[b].sigmoid() >= 0.5).sum()) for b in range(B)]
    assert select_tokens(no, None, 3).get("f2") is None
    # near matrix and the greedy assignment: nearest free node, descending existence order, radius and cosine inclusive, invalid tokens never
    up = np.array([0.0, 1.0])
    gpos, gdir = np.array([[0.0, 0.0], [0.0, 2.0]]), np.tile(up, (2, 1))
    tpos = np.array([[0.0, 1.0], [0.0, 0.2], [0.0, 1.0], [0.0, 3.5]])
    tdir = np.tile(up, (4, 1))
    tdir[2] = [0.0, -1.0]                                                                                   # opposite heading: never near
    near, d = near_matrix(tpos, tdir, np.array([True, True, True, True]), gpos, gdir, 1.5, 0.5)
    assert near.tolist() == [[True, True], [True, False], [False, False], [False, True]], near
    gt, gd = assign_greedy(near, d, 4)
    assert gt.tolist() == [0, -1, -1, 1] and np.allclose(gd[[0, 3]], [1.0, 1.5]) and np.isnan(gd[[1, 2]]).all(), (gt, gd)       # token 0 (the stronger) takes node 0 although token 1 is closer; node 1 goes to token 3 (1.5 = r)
    assert near_matrix(tpos[:1], tdir[:1], np.array([False]), gpos, gdir)[0].sum() == 0 and near_matrix(np.array([[0.0, 1.5001]]), tdir[:1], np.array([True]), gpos[:1], gdir[:1])[0].sum() == 0
    assert near_matrix(np.array([[0.0, 0.0]]), np.array([[0.5, 0.8660254]]), np.array([True]), gpos[:1], gdir[:1])[0].sum() == 1 and near_matrix(np.array([[0.0, 0.0]]), np.array([[0.6, 0.8]]), np.array([True]), gpos[:1], gdir[:1])[0].sum() == 1
    assert near_matrix(np.array([[0.0, 0.0]]), np.array([[0.8, 0.6]]), np.array([True]), gpos[:1], gdir[:1])[0].sum() == 1 and near_matrix(np.array([[0.0, 0.0]]), np.array([[0.9, 0.3]]), np.array([True]), gpos[:1], gdir[:1])[0].sum() == 0
    gt2, _ = assign_greedy(np.array([[True, True]]), np.array([[1.0, 1.0]]), 1)
    assert gt2.tolist() == [0], "a tie in distance goes to the lower GT node"
    # types
    fl = np.zeros((6, 8), bool)
    fl[1, 2] = fl[2, 3] = fl[3, 0] = fl[4, 1] = True                                                         # split, merge, stop, exit
    fl[5, 6] = fl[5, 2] = True                                                                              # win_in + split: fork wins
    assert types_of(fl).tolist() == [1, 2, 3, 4, 5, 2]
    fl2 = np.zeros((2, 8), bool)
    fl2[0, 4] = fl2[1, 7] = True
    assert types_of(fl2).tolist() == [6, 6]
    # a whole frame: g0 -> g1 (short), g1 -> g2 (short), g2 -> g3 (long); g2 and g3 are covered by no token
    def gt_table(pos, links, longs, flags=None):
        M = len(pos)
        t = dict(pos=np.array(pos, float), dirv=np.tile(up, (M, 1)), cell=np.zeros((M, 2), int), flags=np.zeros((M, 8), bool) if flags is None else flags, piece=np.zeros(M, int), sstart=np.zeros(M, bool),
                 src=np.array([a for a, _ in links], int), dst=np.array([b for _, b in links], int), long=np.array(longs, bool), conn=np.array([bool(g) for g in longs]), scale=np.full((len(links), 2), np.nan))
        t["sstart"][:1] = True
        for l, g in enumerate(longs):
            if g:
                t["scale"][l] = [0.3, 0.4]
        return t

    fl3 = np.zeros((4, 8), bool)
    fl3[1, 0] = True                                                                                         # g1 = stop
    table = gt_table([[0, 0], [0, 4], [0, 8], [0, 30]], [(0, 1), (1, 2), (2, 3)], [False, False, True], fl3)
    tk = dict(pos=np.array([[0, 0.3], [0, 0.9], [0, 4.1], [0, 8.0], [0, 8.2]], float), dirv=np.array([[0, 1], [0, 1], [0, 1], [0, -1], [0, 1]], float), valid=np.array([True, True, True, True, False]),
              e_p=np.array([0.9, 0.8, 0.7, 0.6, 0.4]))
    allowed = np.zeros((5, 5), bool)
    allowed[0, 2] = True
    lab = label_frame(tk, table, K=5, allowed=allowed, ks=(2, 5))
    assert lab["gt_id"].tolist() == [0, -1, 1, -1, -1] and lab["gt_type"].tolist() == [1, 0, 4, 0, 0] and lab["gt_aux"][0].tolist() == [True, False, False] and lab["gt_flags"][2, 0]
    assert lab["n_gt"] == 4 and lab["n_gt_assigned"] == 2 and lab["n_links"] == 3 and lab["n_links_c"] == 1 and lab["gt_out"] == 2 and lab["n_nosrc"] == 0 and lab["n_reach"] == 1 and lab["n_pairs"] == 1
    assert lab["ypair"][0].tolist() == [0, 2] and lab["ypair"][1].tolist() == [-1, -1] and not lab["ypair_long"][0] and lab["ypair_gate"][0] and np.isnan(lab["ypair_scale"][0]).all()
    assert lab["cov_nodes_at"].tolist() == [1, 2] and lab["cov_links_at"].tolist() == [0, 1] and lab["n_valid"] == 4 and lab["n_deliv"] == 4 and lab["n_deliv_near"] == 3 and lab["n_gt_deliv_cov"] == 2
    lab = label_frame(tk, table, K=3, allowed=None, ks=(2, 5))                                               # K = 3: the first three tokens only; the stats still see five
    assert lab["gt_id"].shape == (3,) and lab["n_pairs"] == 1 and not lab["ypair_gate"][0] and lab["n_reach"] == 0 and lab["cov_nodes_at"].tolist() == [1, 2]
    # contraction: g0 -> g1 -> g2 with g1 uncovered becomes one link g0 -> g2 (chord 10 m: long, no fitted scale); tokens near g2: two of them, both are targets of the source token
    table2 = gt_table([[0, 0], [0, 5], [0, 10]], [(0, 1), (1, 2)], [False, False])
    tk2 = dict(pos=np.array([[0, 0.1], [0, 10.2], [0, 9.0], [5, 5]], float), dirv=np.array([[0, 1]] * 4, float), valid=np.ones(4, bool), e_p=np.array([0.9, 0.8, 0.7, 0.6]))
    lab = label_frame(tk2, table2, K=4)
    assert lab["gt_id"].tolist() == [0, 2, -1, -1] and lab["n_links_c"] == 1 and lab["n_pairs"] == 2 and lab["ypair"][:2].tolist() == [[0, 1], [0, 2]] and lab["ypair_long"][:2].tolist() == [True, True]
    assert np.isnan(lab["ypair_scale"][:2]).all() and lab["gt_out"] == 2 and not lab["ypair_gate"][:2].any()
    lab = label_frame(tk2, table2, K=4, allowed=np.ones((4, 4), bool))
    assert lab["ypair_gate"][:2].all() and lab["n_reach"] == 1
    # a direct long link keeps its fitted scale; a link without a source token is counted
    table3 = gt_table([[0, 0], [0, 12]], [(0, 1)], [True])
    tk3 = dict(pos=np.array([[0, 0.0], [0, 12.0]], float), dirv=np.array([[0, 1]] * 2, float), valid=np.ones(2, bool), e_p=np.array([0.9, 0.8]))
    lab = label_frame(tk3, table3, K=2)
    assert lab["ypair"][0].tolist() == [0, 1] and lab["ypair_long"][0] and np.allclose(lab["ypair_scale"][0], [0.3, 0.4])
    tk3b = dict(pos=np.array([[0, 12.0], [0, 0.0]], float), dirv=np.array([[0, 1]] * 2, float), valid=np.ones(2, bool), e_p=np.array([0.9, 0.8]))
    lab = label_frame(dict(tk3b, valid=np.array([True, True])), gt_table([[0, 0], [0, 12]], [(1, 0)], [True]), K=2)         # the link runs g1 -> g0: source token = the one near g1 (token 0), target token 1
    assert lab["ypair"][0].tolist() == [0, 1], "link direction: source token -> target token"
    lab = label_frame(dict(tk3, valid=np.array([False, True])), table3, K=2)
    assert lab["n_nosrc"] == 0 and lab["n_links_c"] == 0 and lab["n_pairs"] == 0, "an uncovered end contracts the link away"
    # the frame without GT nodes
    empty = gt_table(np.zeros((0, 2)), [], [])
    lab = label_frame(tk3, empty, K=2)
    assert (lab["gt_id"] == -1).all() and lab["n_gt"] == 0 and lab["n_pairs"] == 0 and lab["cov_nodes_at"].sum() == 0
    # the dump comparison on a toy cache: equal passes, any difference fails
    class Toy:
        K, rows = 4, [7]
        f = dict(cell=np.array([[53, 5, 1224, 9]], np.int16), e_p=np.array([[0.9, 0.6, 0.3, 0.01]], np.float32), pos=np.array([[[-47.0, -43.0], [-49.5, -39.0], [-1.0, -1.0], [0.0, 0.0]]], np.float32),
                 dirv=np.array([[[1.0, 0], [0, 1.0], [1.0, 0], [1.0, 0]]], np.float32), flags_p=np.full((1, 4, 8), 0.25, np.float32), s_p=np.full((1, 4), 0.5, np.float32))
    toy = Toy()

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
        assert not ok and rep["worst"][key] > 0, (key, rep)
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
    assert nd["cell"].tolist() == [[1, 3], [24, 24]] and nd["section"].tolist() == [0, 0] and np.allclose(nd["e"], [0.9, 0.6])
    assert cache_name("e292_a_s0", 384, "9ad1d4ce30") == "be292_a_0_K384_9ad1d4ce30_v1" and cache_name("e273_p_s1", 512, "x") == "be273_p_1_K512_x_v1"
    print("lane_tokcache selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("build")
    s.add_argument("--run", required=True)
    s.add_argument("--split", required=True, choices=("val", "val_clean", "train"))
    s.add_argument("--K", type=int, default=384)
    s.add_argument("--no-f2", dest="f2", action="store_false")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--workers", type=int, default=2)
    s = sub.add_parser("fixture")
    s.add_argument("--cache", required=True)
    s.add_argument("--dump", required=True)
    s.add_argument("--prefix", action="store_true", help="the cache is a limited build: compare the first frames of the dump only")
    s = sub.add_parser("report")
    s.add_argument("--cache", required=True)
    s.add_argument("--write", action="store_true")
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "build":
        build(a)
    elif a.cmd == "report":
        rep_ = report(_abs(a.cache))
        print(json.dumps(rep_, indent=1))
        if a.write:
            p_ = os.path.join(_abs(a.cache), "manifest.json")
            _atomic_json(p_, dict(json.load(open(p_)) if os.path.exists(p_) else json.load(open(os.path.join(_abs(a.cache), "meta.json"))), numbers=rep_))
    else:
        c = TokCache(_abs(a.cache))
        ok, rep = compare_dump(c, pickle.load(open(_abs(a.dump), "rb")), prefix=a.prefix)
        print(("FIXTURE PASS " if ok else "FIXTURE FAIL ") + json.dumps(rep))
        sys.exit(0 if ok else 1)
