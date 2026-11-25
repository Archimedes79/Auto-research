"""E395 GT geometry of the token frames: derived data under runs/e395_gtgeom (not in git), the labels of the position head's curve and node loss (`net/model/jtoken.curve_losses`); no model input reads them.

    wsl bash tools/env/_py14.sh lane_gtgeom_data.py build --split train|val --seed 0      # -> runs/e395_gtgeom/<split>_s<seed>.npz (never overwritten: written under a temp name, hard-linked to the final name, the temp name removed)
    wsl bash tools/env/_py14.sh lane_gtgeom_data.py selftest
    g = load(split, seed, cache_id, tokens)                                                # the arrays, checked against the cache; `lane_ptok_data.PTokData(geom=True)` slices them per frame

Per frame of the cache of B' (`lane_ptok_data.PTokData(split, seed, "e292_a")`, the cache order) from the E357 GT table of the frame (`lane_tok_data.TokData(split, res=2.0, feats=False).table(i)`: pos, dirv, src, dst) and the token labels
of the cache (gt_id, gd, valid): gt_pos / gt_dir (the table's pos / dirv), gt_btok[g] = the valid token k with gt_id[k] == g and the smallest finite gd (ties: the smaller k), -1 none; gl = one row per GT link (u, v, pre, suc): u -> v the table's link
(src, dst), pre = the GT predecessor w of u (a link w -> u, w != v) and suc = the GT successor x of v (a link v -> x, x != u) that continue most straight: the smallest turning angle between (G_u - G_w) and (G_v - G_u), and between (G_x - G_v) and
(G_v - G_u); -1 when there is none, the first candidate in the table's link order wins a tie. The file holds the frames padded to the largest count (gt_pos / gt_dir float32 (F, Gm, 2), gt_btok int32 (F, Gm) -1, gl int32 (F, Lm, 4) -1) with the counts
n_g / n_l, the frame tokens (the order check), the cache's folder name (`cache_id`: ident, seed, K, GT signature, version: a rebuilt cache makes the file stale) and K. Tests: `selftest`, test_lane_tokcache.py
"""
import argparse
import math
import os
import tempfile

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GEOM = os.path.join(ROOT, "runs", "e395_gtgeom")
IDENT = "e292_a"                                                            # the frozen head whose tokens are cached: the labels are made for B' only


def cache_id(dir):
    """the folder name of the cache a `PTokData` split directory belongs to (<cache root>/<this>/<split>)"""
    return os.path.basename(os.path.dirname(os.path.abspath(dir)))


def best_tokens(gt_id, gd, valid, n_g):
    """(n_g,) int: per GT node the valid token k with gt_id[k] == g and the smallest finite gd (ties: the smaller k), -1 none"""
    bt, best = -np.ones(n_g, int), np.full(n_g, np.inf)
    for k in range(len(gt_id)):                                             # ascending k: a strict '<' keeps the smaller k on a tie
        g = int(gt_id[k])
        if valid[k] and g >= 0 and np.isfinite(gd[k]) and gd[k] < best[g]:
            bt[g], best[g] = k, gd[k]
    return bt


def straightest(cands, d):
    """cands = [(vector, node)]: the node whose vector has the smallest turning angle to d (the first one on a tie), -1 for none"""
    best, node = math.inf, -1
    for vec, w in cands:
        a = math.atan2(abs(vec[0] * d[1] - vec[1] * d[0]), vec[0] * d[0] + vec[1] * d[1])
        if a < best:
            best, node = a, int(w)
    return node


def frame_geom(table, gt_id, gd, valid):
    """one frame: the GT table (pos, dirv (M, 2), src, dst (L,)) and the token labels (gt_id, gd, valid (K,)) -> dict(gt_pos (M, 2) f32, gt_dir (M, 2) f32, gt_btok (M,) i32, gl (L, 4) i32): module docstring"""
    pos, dirv = np.asarray(table["pos"], float), np.asarray(table["dirv"], float)
    src, dst = (np.asarray(table[k], int) for k in ("src", "dst"))
    M = len(pos)
    into, outof = [[] for _ in range(M)], [[] for _ in range(M)]
    for u, v in zip(src, dst):                                              # the link order of the table
        into[v].append(int(u))
        outof[u].append(int(v))
    gl = -np.ones((len(src), 4), int)
    for l, (u, v) in enumerate(zip(src, dst)):
        d = pos[v] - pos[u]
        gl[l] = (u, v, straightest([(pos[u] - pos[w], w) for w in into[u] if w != v], d), straightest([(pos[x] - pos[v], x) for x in outof[v] if x != u], d))
    return dict(gt_pos=pos.astype(np.float32), gt_dir=dirv.astype(np.float32), gt_btok=best_tokens(gt_id, gd, valid, M).astype(np.int32), gl=gl.astype(np.int32))


def pack(frames, tokens, cache, K):
    """the per-frame dicts of `frame_geom` -> the arrays of the file (padded to the largest count, at least 1) + n_g, n_l, tokens, cache, K"""
    F = len(frames)
    n_g, n_l = (np.array([len(f[k]) for f in frames], np.int32) for k in ("gt_pos", "gl"))
    Gm, Lm = max(1, int(n_g.max(initial=0))), max(1, int(n_l.max(initial=0)))
    a = dict(gt_pos=np.zeros((F, Gm, 2), np.float32), gt_dir=np.zeros((F, Gm, 2), np.float32), gt_btok=-np.ones((F, Gm), np.int32), gl=-np.ones((F, Lm, 4), np.int32))
    for i, f in enumerate(frames):
        for k in a:
            a[k][i, :len(f[k])] = f[k]
    return dict(a, n_g=n_g, n_l=n_l, tokens=np.array(list(tokens), dtype=str), cache=np.array(cache), K=np.array(int(K)))


def _path(split, seed, root=None):
    return os.path.join(root or GEOM, f"{split}_s{int(seed)}.npz")


def save(arrays, split, seed, root=None):
    """-> the path; written under a temp name, then hard-linked to the final name (`os.link` fails when the name exists, also when another builder made it in the meantime) and the temp name removed: a file that exists is never overwritten (AssertionError)"""
    path = _path(split, seed, root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    np.savez(tmp, **arrays)
    try:
        os.link(tmp, path)
    except FileExistsError:
        raise AssertionError(f"{path} exists: it is never overwritten (move it away to rebuild)") from None
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def load(split, seed, cache, tokens, root=None):
    """the arrays of the file of the split / seed, after the checks that it was made from this cache (`cache_id`) and lists its frames in this order (tokens = the cache's frame tokens)"""
    with np.load(_path(split, seed, root)) as z:
        g = {k: z[k] for k in z.files}
    assert str(g["cache"]) == cache, f"the GT geometry of {split} seed {seed} was made from the cache {g['cache']}, not {cache}: rebuild it"
    assert g["tokens"].tolist() == list(tokens), f"the GT geometry of {split} seed {seed} lists other frames (or another order) than the cache"
    return g


def build(split, seed, root=None, log=print):
    """the file of the split / seed from the real cache of B' and the E357 tables (module docstring; heavy imports here) -> its path"""
    import lane_ptok_data as LP
    import lane_tok_data as LT
    assert not os.path.exists(_path(split, seed, root)), f"{_path(split, seed, root)} exists: it is never overwritten (move it away to rebuild)"          # fail before the work; `save` is the guard
    ds, td = LP.PTokData(split, seed, IDENT), LT.TokData(split, res=2.0, feats=False)
    assert list(ds.tokens) == list(td.tokens), "the cache and the GT tables list other frames (or another order)"
    frames = []
    for i in range(len(ds)):
        it = ds[i]
        frames.append(frame_geom(td.table(i), it["label"]["gt_id"], it["label"]["gd"], it["valid"]))
        if log and ((i + 1) % 500 == 0 or i + 1 == len(ds)):
            log(f"   {split} s{seed}: {i + 1}/{len(ds)} frames", flush=True)
    return save(pack(frames, ds.tokens, cache_id(ds.dir), ds.K), split, seed, root)


# ------------------------------------------------------------------------------------------------------------------------------ test
def _toy_table(swap=False):
    """15 GT nodes, 12 links: a chain 0-1-2 with a fork at 2 (3 straight on, 4 turned by 45 degrees), a two-cycle 5 <-> 6, a tie (8 and 9 enter 7 at 45 degrees to either side of 7 -> 10), a straight and a turned predecessor (13, 12 -> 11 -> 14);
    swap: the links 8 -> 7 and 9 -> 7 in the other order"""
    pos = np.array([(0, 0), (2, 0), (4, 0), (6, 0), (6, 2), (10, 0), (12, 0), (22, 0), (20, 2), (20, -2), (24, 0), (30, 0), (28, 0), (28, -2), (32, 0)], float)
    links = [(0, 1), (1, 2), (2, 3), (2, 4), (5, 6), (6, 5), (8, 7), (9, 7), (7, 10), (13, 11), (12, 11), (11, 14)]
    if swap:
        links[6], links[7] = links[7], links[6]
    return dict(pos=pos, dirv=np.tile([1.0, 0.0], (len(pos), 1)), src=np.array([a for a, _ in links]), dst=np.array([b for _, b in links]))


def selftest():
    # the best token of a GT node: the smallest finite gd of the VALID tokens of it, the smaller token on a tie
    gt_id, gd = np.array([0, 0, 1, 2, 3, 3, -1, 5]), np.array([0.3, 0.1, 0.5, np.nan, 0.2, 0.2, np.nan, 0.4])
    valid = np.array([True, True, False, True, True, True, False, True])
    want_bt = [1, -1, -1, 4, -1, 7] + [-1] * 9                              # node 0: tokens 0 / 1 -> 1; 1: its token is invalid; 2: gd NaN; 3: a tie of tokens 4, 5 -> 4; 5: token 7
    assert best_tokens(gt_id, gd, valid, 15).tolist() == want_bt
    # the ghost nodes by hand (rows (u, v, pre, suc))
    t = _toy_table()
    f = frame_geom(t, gt_id, gd, valid)
    want = [(0, 1, -1, 2), (1, 2, 0, 3), (2, 3, 1, -1), (2, 4, 1, -1), (5, 6, -1, -1), (6, 5, -1, -1), (8, 7, -1, 10), (9, 7, -1, 10), (7, 10, 8, -1), (13, 11, -1, 14), (12, 11, -1, 14), (11, 14, 12, -1)]
    # (1, 2): the fork at 2 -> 3 straight on beats 4 (45 degrees); (5, 6) / (6, 5): the other direction of the pair is no ghost; (7, 10): 8 and 9 turn by 45 degrees alike, the first of the table wins; (11, 14): 12 (straight) beats 13 (45 degrees)
    assert f["gl"].tolist() == [list(r) for r in want], f["gl"].tolist()
    assert f["gt_btok"].tolist() == want_bt and f["gt_pos"].dtype == np.float32 and f["gl"].dtype == np.int32 and np.array_equal(f["gt_pos"], t["pos"]) and np.array_equal(f["gt_dir"], t["dirv"])
    assert frame_geom(_toy_table(True), gt_id, gd, valid)["gl"][8].tolist() == [7, 10, 9, -1], "the first candidate wins the tie (the table's link order)"
    # the file: padded frames, counts, the checks against the cache, never overwritten
    small = frame_geom(dict(pos=t["pos"][:2], dirv=t["dirv"][:2], src=np.array([0]), dst=np.array([1])), np.array([-1]), np.array([np.nan]), np.array([True]))
    empty = frame_geom(dict(pos=np.zeros((0, 2)), dirv=np.zeros((0, 2)), src=np.zeros(0, int), dst=np.zeros(0, int)), np.array([-1]), np.array([np.nan]), np.array([False]))
    a = pack([f, small, empty], ["ta", "tb", "tc"], "cache_x", 512)
    assert a["gt_pos"].shape == (3, 15, 2) and a["gl"].shape == (3, 12, 4) and a["n_g"].tolist() == [15, 2, 0] and a["n_l"].tolist() == [12, 1, 0] and int(a["K"]) == 512
    assert a["gl"][1, 0].tolist() == [0, 1, -1, -1] and (a["gl"][1, 1:] == -1).all() and (a["gt_btok"][1] == -1).all() and (a["gt_pos"][1, 2:] == 0).all() and (a["gl"][2] == -1).all() and (a["gt_btok"][0, :15] == want_bt).all()
    assert pack([empty], ["t"], "c", 1)["gl"].shape == (1, 1, 4), "a split of frames without GT keeps one padded row"
    with tempfile.TemporaryDirectory() as root:
        path = save(a, "val", 3, root)
        assert os.path.basename(path) == "val_s3.npz" and not [f_ for f_ in os.listdir(root) if "tmp" in f_]
        g = load("val", 3, "cache_x", ["ta", "tb", "tc"], root)
        assert all(np.array_equal(g[k], a[k]) for k in a if k not in ("cache",))
        for bad in (dict(cache="cache_y", tokens=["ta", "tb", "tc"]), dict(cache="cache_x", tokens=["tb", "ta", "tc"]), dict(cache="cache_x", tokens=["ta", "tb"])):
            try:
                load("val", 3, bad["cache"], bad["tokens"], root)
            except AssertionError:
                continue
            raise RuntimeError(f"a stale file was accepted: {bad}")
        before = open(path, "rb").read()
        other = dict(a, gt_btok=a["gt_btok"] + 1)                           # another builder's file for the same name: refused (os.link fails on an existing name, whatever was checked before), the file stays as it was, no temp file is left
        try:
            save(other, "val", 3, root)
        except AssertionError:
            pass
        else:
            raise RuntimeError("an existing file was overwritten")
        assert open(path, "rb").read() == before and os.listdir(root) == ["val_s3.npz"], os.listdir(root)
    assert cache_id(os.path.join("x", "bz_0_K512_sig_v1", "val")) == "bz_0_K512_sig_v1"
    print("lane_gtgeom_data selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "selftest"))
    ap.add_argument("--split", choices=("train", "val"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    else:
        assert a.split, "--split"
        print(build(a.split, a.seed))
