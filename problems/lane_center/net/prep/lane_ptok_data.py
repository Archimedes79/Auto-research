"""E365 loader of the predicted-node cache (`lane_tokcache.py`, spec: section 2a of `experiments/E365_e2e-token-champion-s1/experiment.md`): the tokens of the frozen lane head B' (or B) of a frame and the GT labels of the tokens.

    ds, dl = loader("train" | "val", seed=0, bs=8, workers=4, token_drop=0.0, ident="e292_a")       # the ONE finished cache of B' seed 0 under runs/e365_cache
    batch["tok"] (B, K, 64) f16, ["pos"], ["dirv"] (B, K, 2), ["e"], ["e_raw"], ["s"], ["uout"], ["uin"] (B, K), ["flags"] (B, K, 8) logits, ["valid"] (B, K) bool,
    ["ytype"] (B, K) int64 (0 none .. 6 boundary), ["y"] (B, K, K) uint8 (1 where token j is in G(i)), ["ylong"] (B, K, K) bool = y and the chord |pos_j - pos_i| of the TOKENS above 8 m (not the chord of the GT link), ["yscale"] (B, K, K, 2) NaN outside the long GT links with a fit, ["gt_out"] (B,) the GT links lost,
    ["token"], ["row"], ["tokens_np"] (a list of per-frame dicts, float64 pos / dirv / e_p / flags_p / s_p, int cell, bool valid: what the read-out writer needs; NO GT field), ["label"] (the GT-side fields
    gt_id, gd, gt_flags, gt_aux for the controls and diagnostics: the model must not read them)
    to_pred(tokens_np[b], keep, e_node, links)                                                                       # the qformat prediction of a frame from tokens + the head's output, no GT field
Token drop: `token_drop` > 0 removes each token with that probability by a generator seeded with (the loader's seed, the frame's token) (`lane_tok_data.keep_nodes`): a removed token is not valid, has type 0 and no target
pair (the contraction of the targets is NOT recomputed: this is for the degraded read-out only).
Tests: net/tests/test_lane_tokcache.py
"""
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub_ in ("training/data", "net/prep", "net/model", "net/loss", "evaluation"):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import lane_tokcache as LC  # noqa: E402
import qtargets as QT  # noqa: E402

TOKEN_FIELDS = ("tok", "pos", "dirv", "e", "e_raw", "s", "uout", "uin", "flags", "valid")


def cache_dir(split, seed, ident="e292_a"):
    """the directory of the cache of run `ident`_s`seed` for the split under `lane_tokcache.CACHE`: exactly one finished cache (one K) must exist"""
    import lane_tok_data as LT
    sig = os.path.basename(LT.LD.lane_store()[1])
    pat = os.path.join(LC.CACHE, f"b{ident}_{seed}_K*_{sig}_{LC.VERSION}", split)
    hits = [h for h in sorted(glob.glob(pat)) if os.path.exists(os.path.join(h, "manifest.json"))]
    assert len(hits) == 1, f"{len(hits)} finished caches match {pat}: {hits}"
    return hits[0]


def densify(ypair, ylong, yscale, K, keep=None):
    """the sparse target pairs of a frame (ypair (P, 2) int, -1 padded) -> y (K, K) uint8, ylong (K, K) bool, yscale (K, K, 2) float32 (NaN where there is no fitted scale); pairs with a removed token (keep (K,) bool) are left out"""
    y, yl, ys = np.zeros((K, K), np.uint8), np.zeros((K, K), bool), np.full((K, K, 2), np.nan, np.float32)
    ok = ypair[:, 0] >= 0
    i, j = ypair[ok, 0].astype(int), ypair[ok, 1].astype(int)
    if keep is not None:
        k = keep[i] & keep[j]
        i, j, ok_idx = i[k], j[k], np.flatnonzero(ok)[k]
    else:
        ok_idx = np.flatnonzero(ok)
    y[i, j], yl[i, j], ys[i, j] = 1, ylong[ok_idx], yscale[ok_idx]
    return y, yl, ys


class PTokData(torch.utils.data.Dataset):
    def __init__(self, split="train", seed=0, ident="e292_a", token_drop=0.0, dir=None):
        assert split in ("train", "val"), split
        self.dir = dir or cache_dir(split, seed, ident)
        self.c = LC.TokCache(self.dir)
        assert self.c.done == len(self.c), f"{self.dir} is not finished ({self.c.done} of {len(self.c)} frames)"
        self.K, self.seed, self.drop = self.c.K, int(seed), float(token_drop)
        self.rows, self.tokens = self.c.rows, self.c.tokens

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        f = self.c.f
        keep = np.ones(self.K, bool)
        if self.drop > 0:
            from lane_tok_data import keep_nodes
            keep = keep_nodes(self.tokens[i], self.K, self.drop, self.seed)
        valid = np.array(f["valid"][i]) & keep
        y, _, ys = densify(np.array(f["ypair"][i]), np.array(f["ypair_long"][i]), np.array(f["ypair_scale"][i]), self.K, keep if self.drop > 0 else None)
        pos = np.array(f["pos"][i], np.float64)
        yl = chord_long(y, pos)                  # the class of a pair = the chord of its TOKENS (> 8 m); the class of the GT link is in the cache (`ypair_long`)
        out = {n: np.array(f[n][i]) for n in TOKEN_FIELDS if n in f}
        out.update(valid=valid, ytype=np.where(keep, np.array(f["gt_type"][i]), 0).astype(np.int64), y=y, ylong=yl, yscale=ys, gt_out=int(f["gt_out"][i]), token=self.tokens[i], row=int(self.rows[i]),
                   tokens_np=dict(pos=np.array(f["pos"][i], np.float64), dirv=np.array(f["dirv"][i], np.float64), e_p=np.array(f["e_p"][i], np.float64), flags_p=np.array(f["flags_p"][i], np.float64),
                                  s_p=np.array(f["s_p"][i], np.float64), cell=np.array(f["cell"][i]).astype(np.int64), valid=valid.copy()),
                   label=dict(gt_id=np.array(f["gt_id"][i]), gd=np.array(f["gd"][i]), gt_flags=np.array(f["gt_flags"][i]), gt_aux=np.array(f["gt_aux"][i])))
        return out


def collate(b):
    out = {n: torch.from_numpy(np.stack([o[n] for o in b])) for n in TOKEN_FIELDS + ("ytype", "y", "ylong", "yscale") if n in b[0]}
    out.update(gt_out=torch.tensor([o["gt_out"] for o in b]), token=[o["token"] for o in b], row=torch.tensor([o["row"] for o in b]), tokens_np=[o["tokens_np"] for o in b],
               label={n: torch.from_numpy(np.stack([o["label"][n] for o in b])) for n in b[0]["label"]})
    return out


def loader(split="train", seed=0, bs=8, workers=4, token_drop=0.0, ident="e292_a", shuffle=None, dir=None):
    shuffle = split == "train" if shuffle is None else shuffle
    ds = PTokData(split, seed, ident, token_drop, dir)
    return ds, torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers, collate_fn=collate, drop_last=shuffle, persistent_workers=workers > 0,
                                           prefetch_factor=2 if workers > 0 else None)


def to_pred(frame, keep, e_node, links, flags=None):
    """the qformat prediction (`qformat.py`) of one frame from the cached tokens `frame` (an entry of `tokens_np`), the kept tokens `keep` (K,) bool (the head's p_none < 0.5), the node existence `e_node` (K,) (= 1 - p_none)
    and the head's links on TOKEN ids (dict src, dst, p, long, a, b); flags (K, 8) probabilities (default: those of the cached B), s = B''s section-start probability, every node in section 0 (the section-free read-out of
    E357), links with an end that is not kept are dropped and the others renumbered to the kept nodes; no assignment, no maps; nothing of the GT table enters"""
    keep = np.asarray(keep, bool) & np.asarray(frame["valid"], bool)                                              # a padding token is never a node
    idx = np.flatnonzero(keep)
    new = -np.ones(len(keep), int)
    new[idx] = np.arange(len(idx))
    fl = np.asarray(frame["flags_p"] if flags is None else flags, float)
    c = np.asarray(frame["cell"], int)[idx]
    nodes = dict(pos=np.asarray(frame["pos"], float)[idx], dirv=np.asarray(frame["dirv"], float)[idx], e=np.asarray(e_node, float)[idx], flags=fl[idx], s=np.asarray(frame["s_p"], float)[idx],
                 cell=np.stack([c // LC.N, c % LC.N], 1), section=np.zeros(len(idx), int))
    src, dst = np.asarray(links["src"], int), np.asarray(links["dst"], int)
    ok = (new[src] >= 0) & (new[dst] >= 0) & (src != dst)
    pick = lambda k: np.asarray(links[k])[ok]  # noqa: E731
    lk = dict(src=new[src[ok]], dst=new[dst[ok]], p=pick("p").astype(float), long=pick("long").astype(bool), a=pick("a").astype(float), b=pick("b").astype(float), curves=None)
    return dict(nodes=nodes, links=lk, assign=None, maps=None)


def toy_cache_test():
    """the loader on a tiny cache written here: shapes, dense targets, the seeded token drop"""
    import json
    import tempfile
    F, K = 3, 32
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        arr = dict(tok=rng.random((F, K, 4)).astype(np.float16), pos=rng.random((F, K, 2)).astype(np.float32), dirv=rng.random((F, K, 2)).astype(np.float32), e=rng.random((F, K)).astype(np.float32),
                   e_raw=rng.random((F, K)).astype(np.float32), s=rng.random((F, K)).astype(np.float32), uout=rng.random((F, K)).astype(np.float32), uin=rng.random((F, K)).astype(np.float32),
                   flags=rng.random((F, K, 8)).astype(np.float32), valid=np.ones((F, K), bool), gt_type=np.tile((np.arange(K) % 7).astype(np.int8), (F, 1)), gt_out=np.arange(F, dtype=np.int32) + 5,
                   ypair=-np.ones((F, 8, 2), np.int16), ypair_long=np.zeros((F, 8), bool), ypair_scale=np.full((F, 8, 2), np.nan, np.float32), e_p=rng.random((F, K)).astype(np.float32),
                   flags_p=rng.random((F, K, 8)).astype(np.float32), s_p=rng.random((F, K)).astype(np.float32), cell=np.tile(np.arange(K, dtype=np.int16), (F, 1)), gt_id=-np.ones((F, K), np.int32),
                   gd=np.full((F, K), np.nan, np.float32), gt_flags=np.zeros((F, K, 8), bool), gt_aux=np.zeros((F, K, 3), bool))
        arr["ypair"][:, :3] = [[0, 2], [0, 3], [4, 1]]
        arr["pos"][:] = 0.0
        arr["pos"][:, 2], arr["pos"][:, 3], arr["pos"][:, 1] = [0.0, 5.0], [0.0, 20.0], [0.0, 9.0]              # token chords: (0, 2) 5 m, (0, 3) 20 m, (4, 1) 9 m
        arr["ypair_long"][:, 0] = True                                                                            # the GT class is the opposite of the token class for (0, 2) and (0, 3)
        arr["ypair_scale"][:, 1] = [0.3, 0.4]
        arr["valid"][1, 30:] = False
        for n_, v in arr.items():
            np.save(os.path.join(d, n_ + ".npy"), v)
        json.dump(dict(rows=[10, 11, 12], tokens=["ta", "tb", "tc"]), open(os.path.join(d, "frames.json"), "w"))
        json.dump(dict(done=F), open(os.path.join(d, "progress.json"), "w"))
        ds = PTokData("val", dir=d)
        assert len(ds) == 3 and ds.K == K
        b = collate([ds[0], ds[1]])
        assert b["tok"].shape == (2, K, 4) and b["y"].shape == (2, K, K) and b["yscale"].shape == (2, K, K, 2) and b["ytype"].dtype == torch.int64 and b["gt_out"].tolist() == [5, 6] and b["token"] == ["ta", "tb"]
        assert b["row"].tolist() == [10, 11] and int(b["y"][0].sum()) == 3 and b["y"][0, 4, 1] == 1 and not bool(b["valid"][1, 31]) and bool(b["valid"][0, 31])
        assert bool(b["ylong"][0, 0, 3]) and bool(b["ylong"][0, 4, 1]) and not bool(b["ylong"][0, 0, 2]) and int(b["ylong"][0].sum()) == 2, "ylong = the chord of the tokens"
        assert not b["ylong"][0, 3, 0] and np.allclose(b["yscale"][0, 0, 3], [0.3, 0.4]) and np.isnan(b["yscale"][0, 0, 2]).all()
        assert sorted(b["tokens_np"][0]) == ["cell", "dirv", "e_p", "flags_p", "pos", "s_p", "valid"] and sorted(b["label"]) == ["gd", "gt_aux", "gt_flags", "gt_id"], "no GT field in tokens_np"
        assert b["tokens_np"][0]["pos"].dtype == np.float64 and b["tokens_np"][0]["valid"].shape == (K,)
        dr1, dr2, dr3 = (PTokData("val", seed=s_, token_drop=0.5, dir=d) for s_ in (1, 1, 2))
        o1, o2, o3 = dr1[0], dr2[0], dr3[0]
        assert np.array_equal(o1["valid"], o2["valid"]) and not np.array_equal(o1["valid"], o3["valid"]), "the drop is seeded by (seed, frame token)"
        gone = ~o1["valid"]
        assert gone.sum() > 5 and (o1["ytype"][gone] == 0).all() and not o1["y"][gone].any() and not o1["y"][:, gone].any() and (o1["ytype"][~gone] == arr["gt_type"][0][~gone]).all()
        assert PTokData("val", token_drop=0.0, dir=d)[0]["valid"].all()


def chord_long(y, pos):
    """the long class of the positive pairs of `y` (K, K): the chord of the two TOKENS above QT.SHORT (8 m; the class of the pair, not of the GT link) -> (K, K) bool. The same values as the dense `(y > 0) & (norm(pos[None] - pos[:, None]) > SHORT)` from the pair list only
    (the dense K x K chord was 92 % of a `PTokData` item and 43 % of the S4 label step; measured identical on 125 VAL frames)"""
    yl = np.zeros(y.shape, bool)
    i, j = np.nonzero(y > 0)
    yl[i, j] = (np.linalg.norm(pos[j] - pos[i], axis=-1) > QT.SHORT)
    return yl


def selftest():
    K = 6
    ypair = -np.ones((8, 2), np.int16)
    ypair[:3] = [[0, 2], [0, 3], [4, 1]]
    ylong = np.zeros(8, bool)
    ylong[1] = True
    ys = np.full((8, 2), np.nan, np.float32)
    ys[1] = [0.3, 0.4]
    y, yl, sc = densify(ypair, ylong, ys, K)
    assert y.sum() == 3 and y[0, 2] == 1 and y[0, 3] == 1 and y[4, 1] == 1 and yl[0, 3] and not yl[0, 2] and np.allclose(sc[0, 3], [0.3, 0.4]) and np.isnan(sc[0, 2]).all() and np.isnan(sc[5]).all() and y[2, 0] == 0
    y, yl, sc = densify(ypair, ylong, ys, K, keep=np.array([True, True, False, True, True, True]))                  # token 2 removed: the pair (0, 2) goes, the others stay
    assert y.sum() == 2 and y[0, 2] == 0 and y[0, 3] == 1 and yl[0, 3] and np.allclose(sc[0, 3], [0.3, 0.4])
    frame = dict(pos=np.arange(12, dtype=float).reshape(6, 2), dirv=np.tile([0.0, 1.0], (6, 1)), e_p=np.linspace(0.9, 0.1, 6), flags_p=np.full((6, 8), 0.2), s_p=np.full(6, 0.3), cell=np.array([53, 5, 1224, 9, 10, 11]),
                 valid=np.ones(6, bool), gt_id=np.arange(6), piece=np.arange(6))
    keep = np.array([True, False, True, True, False, False])
    links = dict(src=[0, 0, 2, 3, 1], dst=[2, 1, 3, 3, 0], p=[0.9, 0.8, 0.7, 0.6, 0.5], long=[False, False, True, False, False], a=[0.3] * 5, b=[0.4] * 5)
    pr = to_pred(frame, keep, 1 - np.arange(6) / 10, links)
    assert pr["nodes"]["pos"].tolist() == [[0.0, 1.0], [4.0, 5.0], [6.0, 7.0]] and pr["nodes"]["section"].tolist() == [0, 0, 0] and np.allclose(pr["nodes"]["e"], [1.0, 0.8, 0.7]) and pr["nodes"]["cell"].tolist() == [[1, 3], [24, 24], [0, 9]]
    assert pr["links"]["src"].tolist() == [0, 1] and pr["links"]["dst"].tolist() == [1, 2] and pr["links"]["long"].tolist() == [False, True] and pr["links"]["p"].tolist() == [0.9, 0.7], pr["links"]
    assert pr["assign"] is None and pr["maps"] is None and sorted(pr["nodes"]) == ["cell", "dirv", "e", "flags", "pos", "s", "section"], "no GT field in the prediction"
    pr2 = to_pred(dict(frame, gt_id=np.zeros(6, int), piece=np.ones(6, int)), keep, 1 - np.arange(6) / 10, links)
    assert all(np.array_equal(pr["nodes"][k], pr2["nodes"][k]) for k in pr["nodes"]), "a GT field changed the prediction"
    pr3 = to_pred(frame, keep, 1 - np.arange(6) / 10, links, flags=np.full((6, 8), 0.7))
    assert np.allclose(pr3["nodes"]["flags"], 0.7)
    pr4 = to_pred(dict(frame, valid=np.array([True, False, False, True, True, True])), np.ones(6, bool), np.ones(6), dict(src=[0, 0], dst=[2, 3], p=[0.5, 0.5], long=[False, False], a=[0.3] * 2, b=[0.4] * 2))
    assert len(pr4["nodes"]["pos"]) == 4 and pr4["links"]["src"].tolist() == [0] and pr4["links"]["dst"].tolist() == [1], "a padding token became a node"
    # the removed SOURCE of a pair goes too (not only a removed target)
    y, yl, sc = densify(ypair, ylong, ys, K, keep=np.array([True, True, True, True, False, True]))
    assert y.sum() == 2 and y[4, 1] == 0 and y[0, 2] == 1
    pos6 = np.array([[0.0, 0.0], [8.0, 0.0], [8.0001, 0.0], [20.0, 0.0], [3.0, 4.0], [0.0, 9.0]])
    y6 = np.zeros((6, 6), np.uint8)
    y6[0, 1] = y6[0, 2] = y6[0, 3] = y6[0, 4] = y6[5, 0] = y6[1, 3] = 1
    want = np.zeros((6, 6), bool)
    want[0, 2] = want[0, 3] = want[5, 0] = want[1, 3] = True                       # a chord of exactly 8 m is short, 8.0001 m long, 5 m short; the pair (1, 3) is 12 m; a pair that is no positive is never long
    assert np.array_equal(chord_long(y6, pos6), want), chord_long(y6, pos6)
    assert not chord_long(np.zeros((6, 6), np.uint8), pos6).any()
    dense = (y6 > 0) & (np.linalg.norm(pos6[None, :, :] - pos6[:, None, :], axis=-1) > QT.SHORT)
    assert np.array_equal(chord_long(y6, pos6), dense), "the pair-list form equals the dense form"
    toy_cache_test()
    print("lane_ptok_data selftest ok")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("selftest", "peek"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ident", default="e292_a")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    else:
        ds, dl = loader(a.split, a.seed, bs=4, workers=0, shuffle=False, ident=a.ident)
        b = next(iter(dl))
        print({k: (tuple(v.shape), str(v.dtype)) for k, v in b.items() if torch.is_tensor(v)}, "frames", len(ds), "K", ds.K)
