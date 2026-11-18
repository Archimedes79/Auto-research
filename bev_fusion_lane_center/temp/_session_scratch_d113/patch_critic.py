import io
import os
import sys

W = os.path.join("temp", "_session_scratch_d113", "work")


def patch(name, edits):
    p = os.path.join(W, name)
    s = io.open(p, encoding="utf-8").read()
    for a, b in edits:
        assert s.count(a) == 1, (name, a[:90], s.count(a))
        s = s.replace(a, b)
    io.open(p, "w", encoding="utf-8").write(s)


TOKCACHE = [
    # docstring: what `tok` is, what gt_out counts
    ("node token that B's link head reads), f2 (float16, 144", "node token of the node head: the input of B's link head, which adds its own context blocks on top), f2 (float16, 144"),
    ("n_gt, n_gt_assigned, n_links (GT links), n_links_c (contracted), gt_out (GT links with an end covered by no token), n_nosrc (contracted links without a source token)",
     "n_gt, n_gt_assigned, n_links (GT links), n_links_c (contracted), gt_out (GT links with an end covered by no token: counted BEFORE the contraction, which re-joins a chain through such ends into one link), n_nosrc (contracted\nlinks without a source token: their start node is covered but assigned to no token)"),
    # compare_dump: the order is part of the pass condition (only ties may reorder), a token behind the stored nodes fails
    ("""        order_diff += int(not np.array_equal(cell_d, cell_c))
        if sorted(cell_d.tolist()) != sorted(cell_c.tolist()):
            bad.append((i, "cells differ"))
            continue""", """        order_diff += int(not np.array_equal(cell_d, cell_c))
        if sorted(cell_d.tolist()) != sorted(cell_c.tolist()):
            bad.append((i, "cells differ"))
            continue
        if not np.array_equal(np.asarray(d["e"][:nc], np.float64), f["e_p"][i, :nc].astype(np.float64)):             # other cells at a position are fine only where the existence is equal (a tie)
            bad.append((i, "order differs beyond ties"))
            continue"""),
    ("""        if n < cache.K and nc == n:
            assert not (f["e_p"][i, n:] >= e_min).any(), f"frame {i}: a cache token at or above {e_min} behind the {n} stored nodes"
""", """        if n < cache.K and nc == n and (f["e_p"][i, n:] >= e_min).any():
            bad.append((i, f"a cache token at or above {e_min} behind the {n} stored nodes"))
"""),
    # build: a finished cache is not an error, a resume must match the settings of the start
    ('''    assert done % bs == 0 and done <= n_all, f"progress {done} is not a multiple of {bs} within {n_all}"''',
     '''    assert (done % bs == 0 or done == n_all) and done <= n_all, f"progress {done} is not a multiple of {bs} within {n_all}"
    if done == n_all and not fresh:
        print(f"cache {out} is finished ({n_all} frames)")
        return out
    if not fresh:                                                                                          # a resume continues the SAME build: run checkpoint, budget, features, label code, GT
        meta = json.load(open(os.path.join(out, "meta.json")))
        want = dict(run_sha=_sha(os.path.join(run, "model.pt")), K=a.K, f2=bool(a.f2), version=VERSION, lane_gt=sig, split=split, r=R_NEAR, cos=COS_NEAR, table_version=LT.TABLE_VERSION)
        diff = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
        assert not diff, f"{out} was started with other settings (stored, now): {diff}"'''),
    ('node_stage="mnms" if m.mnms else "plain"', 'node_stage="mnms" if m.mnms else ("soft" if m.soft_nms else "plain")'),
    # fp16 casts must be finite
    ("""        tk = {k: v.cpu().numpy() for k, v in tkt.items()}
        if arrays is None:""", """        tk = {k: v.cpu().numpy() for k, v in tkt.items()}
        for k_ in ("tok", "f2"):
            assert k_ not in tk or np.isfinite(tk[k_]).all(), f"{k_} is not finite in float16 (batch ending at frame {i + len(b['row'])})"
        if arrays is None:"""),
    # report: a non-finite scan of the model inputs
    ("""    nm = c.f["gt_id"][:n] >= 0
    ratio =""", """    nm = c.f["gt_id"][:n] >= 0
    nonfinite = {k: int(sum(int((~np.isfinite(np.asarray(c.f[k][j:j + 500]))).sum()) for j in range(0, n, 500))) for k in ("tok", "f2", "e", "e_raw", "pos", "dirv", "flags", "s", "uout", "uin") if k in c.f}
    ratio ="""),
    ("pairs_max=int(s(\"n_pairs\").max()), pair_class_conflicts=int(s(\"n_conflict\").sum()))", "pairs_max=int(s(\"n_pairs\").max()), pair_class_conflicts=int(s(\"n_conflict\").sum()), nonfinite_values=nonfinite)"),
    ("links_reachable_of_gt_links=round(float(s(\"n_reach\").sum() / max(nl, 1)), 4)", "links_reachable_per_gt_link=round(float(s(\"n_reach\").sum() / max(nl, 1)), 4)"),
    # a determinism fixture: two caches of the same frames equal in every field
    ("""def nodes_of(frame, sel=None):""", """def compare_caches(a, b, n=None):
    \"\"\"two cache directories of the same run and frames (e.g. a full build and a `--limit N` build): the first n frames equal in EVERY field, bit for bit -> (ok, dict field -> number of frames that differ)\"\"\"
    ca, cb = TokCache(a), TokCache(b)
    n = min(len(ca), len(cb)) if n is None else n
    assert ca.rows[:n] == cb.rows[:n] and ca.tokens[:n] == cb.tokens[:n] and set(ca.f) == set(cb.f), "not the same frames or fields"
    diff = {}
    for k in ca.f:
        x, y = np.asarray(ca.f[k][:n]), np.asarray(cb.f[k][:n])
        same = (x == y) | (np.isnan(x) & np.isnan(y)) if x.dtype.kind == "f" else (x == y)
        bad = int((~same.reshape(n, -1).all(1)).sum())
        if bad:
            diff[k] = bad
    return not diff, diff


def nodes_of(frame, sel=None):"""),
    ("""    s = sub.add_parser("report")
    s.add_argument("--cache", required=True)
    s.add_argument("--write", action="store_true")""", """    s = sub.add_parser("report")
    s.add_argument("--cache", required=True)
    s.add_argument("--write", action="store_true")
    s = sub.add_parser("same")
    s.add_argument("--a", required=True)
    s.add_argument("--b", required=True)
    s.add_argument("--n", type=int, default=None)"""),
    ("""    elif a.cmd == "report":""", """    elif a.cmd == "same":
        ok, rep = compare_caches(_abs(a.a), _abs(a.b), a.n)
        print(("SAME " if ok else "DIFFERENT ") + json.dumps(rep))
        sys.exit(0 if ok else 1)
    elif a.cmd == "report":"""),
    ("    wsl bash tools/env/_py14.sh lane_tokcache.py selftest\n    c = TokCache(dir)", "    wsl bash tools/env/_py14.sh lane_tokcache.py same --a <dir> --b <dir of a --limit build> [--n 40]                            # two builds of the same frames equal in every field\n    wsl bash tools/env/_py14.sh lane_tokcache.py selftest\n    c = TokCache(dir)"),
    # selftest additions
    ("""    # the dump comparison on a toy cache: equal passes, any difference fails""", """    # --- after the critic pass: the cases that the first selftest missed
    # a token takes the NEAREST free node, not the lowest index and not the farthest
    gt3, gd3 = assign_greedy(np.array([[True, True]]), np.array([[1.0, 0.7]]), 1)
    assert gt3.tolist() == [1] and np.allclose(gd3, [0.7])
    # the cosine bound is inclusive (cosine exactly 0.5)
    assert near_matrix(np.array([[0.0, 0.0]]), np.array([[0.8660254037844386, 0.5]]), np.array([True]), gpos[:1], gdir[:1])[0].sum() == 1
    # the priority of the types, every pair of flags
    cols = {2: 2, 3: 3, 4: 0, 5: 1, 6: 4}
    for x in cols:
        for y in cols:
            f2_ = np.zeros((1, 8), bool)
            f2_[0, [cols[x], cols[y]]] = True
            assert types_of(f2_)[0] == min(x, y), (x, y)
    # a token beyond K covers nothing and is no target: chain g0 -> g1 -> g2, the only token near g1 is the fourth (K = 3): g1 is uncovered, the contraction joins g0 -> g2 (long, no fitted scale)
    tkc = dict(pos=np.array([[0, 0.1], [0, 10.0], [0, 30.0], [0, 5.0]], float), dirv=np.array([[0, 1]] * 4, float), valid=np.ones(4, bool), e_p=np.array([0.9, 0.8, 0.7, 0.6]))
    lab = label_frame(tkc, table2, K=3, ks=(3, 4))
    assert lab["gt_id"].tolist() == [0, 2, -1] and lab["n_links_c"] == 1 and lab["n_pairs"] == 1 and lab["ypair"][0].tolist() == [0, 1] and lab["ypair_long"][0] and lab["n_nosrc"] == 0
    assert lab["cov_nodes_at"].tolist() == [2, 3] and lab["cov_links_at"].tolist() == [0, 2]
    # a target token beyond K is no target: g0 -> g1, tokens near g1 are the second (inside K) and the fourth (beyond K = 3)
    tkd = dict(pos=np.array([[0, 0.1], [0, 4.0], [0, 30.0], [0, 4.2]], float), dirv=np.array([[0, 1]] * 4, float), valid=np.ones(4, bool), e_p=np.array([0.9, 0.8, 0.7, 0.6]))
    lab = label_frame(tkd, gt_table([[0, 0], [0, 4]], [(0, 1)], [False]), K=3)
    assert lab["n_pairs"] == 1 and lab["ypair"][0].tolist() == [0, 1] and lab["ypair"][1].tolist() == [-1, -1]
    # a covered node that no token is assigned to: the link out of it has no source token
    tk4 = dict(pos=np.array([[0, 0.1], [0, 4.0]], float), dirv=np.array([[0, 1]] * 2, float), valid=np.ones(2, bool), e_p=np.array([0.9, 0.8]))
    lab = label_frame(tk4, gt_table([[0, 0], [0, 0.5], [0, 4]], [(1, 2)], [False]), K=2)
    assert lab["gt_id"].tolist() == [0, 2] and lab["n_nosrc"] == 1 and lab["n_pairs"] == 0 and lab["n_links_c"] == 1
    # the section start / long-link start / long-link end of the assigned node
    lab = label_frame(tk3, table3, K=2)
    assert lab["gt_aux"][0].tolist() == [True, True, False] and lab["gt_aux"][1].tolist() == [False, False, True]
    # a token near both ends of its own link is not its own target
    tk5 = dict(pos=np.array([[0, 0.5], [0, 1.2]], float), dirv=np.array([[0, 1]] * 2, float), valid=np.ones(2, bool), e_p=np.array([0.9, 0.8]))
    lab = label_frame(tk5, gt_table([[0, 0], [0, 1.0]], [(0, 1)], [False]), K=2)
    assert lab["gt_id"].tolist() == [0, 1] and lab["n_pairs"] == 1 and lab["ypair"][0].tolist() == [0, 1]
    # a pair that a short and a long link both claim is kept as long and counted, in either order of the links
    tk6 = dict(pos=np.array([[0, 0.1], [0, 8.25]], float), dirv=np.array([[0, 1]] * 2, float), valid=np.ones(2, bool), e_p=np.array([0.9, 0.8]))
    for order in ((0, 1), (1, 0)):
        links6 = [(0, 1), (0, 2)]
        longs6 = [False, True]
        tb = gt_table([[0, 0], [0, 7.5], [0, 9.0]], [links6[o] for o in order], [longs6[o] for o in order])
        lab = label_frame(tk6, tb, K=2)
        assert lab["n_pairs"] == 1 and lab["ypair"][0].tolist() == [0, 1] and lab["ypair_long"][0] and lab["n_conflict"] == 1, (order, lab["n_pairs"], lab["ypair_long"][0], lab["n_conflict"])
    # the delivered statistics: a token with sigmoid(e) exactly 0.5 is delivered
    lab = label_frame(dict(tk4, e_p=np.array([0.5, 0.49])), gt_table([[0, 0], [0, 4]], [(0, 1)], [False]), K=2)
    assert lab["n_deliv"] == 1 and lab["n_deliv_near"] == 1 and lab["n_gt_deliv_cov"] == 1 and lab["n_e50_in"] == 1
    # select_tokens: the other fields against a loop, the active threshold, the counts of the whole map
    for b in range(B):
        flat = e[b].flatten()
        want = sorted(range(2500), key=lambda c: (-float(flat[c]), c))[:7]
        for k, c in enumerate(want):
            i, j = divmod(c, 50)
            assert torch.equal(t["flags"][b, k], no["flags"][b, :, i, j]) and torch.equal(t["s"][b, k], no["s"][b, i, j]) and torch.equal(t["uout"][b, k], no["uout"][b, i, j]) and torch.equal(t["uin"][b, k], no["uin"][b, i, j])
            assert torch.equal(t["off"][b, k], no["off"][b, :, i, j]) and torch.equal(t["dir_raw"][b, k], no["dir_raw"][b, :, i, j])
    assert t["n_e20"].tolist() == [int((e[b].sigmoid() >= 0.2).sum()) for b in range(B)]
    e2 = torch.full((1, 50, 50), -12.0)
    e2[0, 0, 0], e2[0, 0, 1] = -1.0, -3.5                                                                    # probabilities 0.269 (valid) and 0.029 (padding)
    t2 = select_tokens(dict({k: v[:1] for k, v in no.items()}, e=e2), None, 3)
    assert t2["valid"][0].tolist() == [True, False, False] and t2["cell"][0].tolist()[:2] == [0, 1] and t2["n_e50"].tolist() == [0] and t2["n_e20"].tolist() == [1]
    # the dump comparison on a toy cache: equal passes, any difference fails"""),
    ("""    dd = dump_of(toy)
    dd[0]["pos"][1] = [20.0, 20.0]                                                                           # another cell
    assert not compare_dump(toy, dd)[0]""", """    dd = dump_of(toy)
    dd[0]["pos"][1] = [20.0, 20.0]                                                                           # another cell
    assert not compare_dump(toy, dd)[0]
    tie = Toy()                                                                                              # two tokens with the same existence may stand in either order; a swap of different existences is an error
    tie.f = dict(Toy.f, e_p=np.array([[0.6, 0.6, 0.3, 0.01]], np.float32))
    dd = dump_of(tie)
    swap = lambda dd_: [dict(dd_[0], **{k: dd_[0][k][[1, 0, 2]] for k in ("e", "pos", "dirv", "flags", "s")})]  # noqa: E731
    ok, rep = compare_dump(tie, swap(dd))
    assert ok and rep["frames_with_other_order"] == 1, rep
    ok, rep = compare_dump(toy, swap(dump_of(toy)))
    assert not ok, "a swap of two tokens with different existence passed"
    extra = Toy()                                                                                            # a cache token at e_p >= 0.2 behind the stored nodes
    extra.f = dict(Toy.f, e_p=np.array([[0.9, 0.6, 0.3, 0.5]], np.float32))
    assert not compare_dump(extra, dump_of(extra))[0]"""),
]

PTOK = [
    ('''    keep = np.asarray(keep, bool)
    idx = np.flatnonzero(keep)''', '''    keep = np.asarray(keep, bool) & np.asarray(frame["valid"], bool)                                              # a padding token is never a node
    idx = np.flatnonzero(keep)'''),
    ('''    def __init__(self, split="train", seed=0, ident="e292_a", K=None, token_drop=0.0, cache=LC.CACHE):
        assert split in ("train", "val"), split
        self.dir = cache_dir(split, seed, ident, K, cache)''', '''    def __init__(self, split="train", seed=0, ident="e292_a", K=None, token_drop=0.0, cache=LC.CACHE, dir=None):
        assert split in ("train", "val"), split
        self.dir = dir or cache_dir(split, seed, ident, K, cache)'''),
    ('''def loader(split="train", seed=0, bs=8, workers=4, token_drop=0.0, ident="e292_a", K=None, shuffle=None, cache=LC.CACHE):
    shuffle = split == "train" if shuffle is None else shuffle
    ds = PTokData(split, seed, ident, K, token_drop, cache)''', '''def loader(split="train", seed=0, bs=8, workers=4, token_drop=0.0, ident="e292_a", K=None, shuffle=None, cache=LC.CACHE, dir=None):
    shuffle = split == "train" if shuffle is None else shuffle
    ds = PTokData(split, seed, ident, K, token_drop, cache, dir)'''),
    ('''    pr3 = to_pred(frame, keep, 1 - np.arange(6) / 10, links, flags=np.full((6, 8), 0.7))
    assert np.allclose(pr3["nodes"]["flags"], 0.7)''', '''    pr3 = to_pred(frame, keep, 1 - np.arange(6) / 10, links, flags=np.full((6, 8), 0.7))
    assert np.allclose(pr3["nodes"]["flags"], 0.7)
    pr4 = to_pred(dict(frame, valid=np.array([True, False, False, True, True, True])), np.ones(6, bool), np.ones(6), dict(src=[0, 0], dst=[2, 3], p=[0.5, 0.5], long=[False, False], a=[0.3] * 2, b=[0.4] * 2))
    assert len(pr4["nodes"]["pos"]) == 4 and pr4["links"]["src"].tolist() == [0] and pr4["links"]["dst"].tolist() == [1], "a padding token became a node"
    # the removed SOURCE of a pair goes too (not only a removed target)
    y, yl, sc = densify(ypair, ylong, ys, K, keep=np.array([True, True, True, True, False, True]))
    assert y.sum() == 2 and y[4, 1] == 0 and y[0, 2] == 1
    toy_cache_test()'''),
    ('''def selftest():
    K = 6''', '''def toy_cache_test():
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
        arr["ypair_long"][:, 1] = True
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
        assert b["row"].tolist() == [10, 11] and int(b["y"][0].sum()) == 3 and b["y"][0, 4, 1] == 1 and bool(b["ylong"][0, 0, 3]) and not bool(b["valid"][1, 31]) and bool(b["valid"][0, 31])
        assert sorted(b["tokens_np"][0]) == ["cell", "dirv", "e_p", "flags_p", "pos", "s_p", "valid"] and sorted(b["label"]) == ["gd", "gt_aux", "gt_flags", "gt_id"], "no GT field in tokens_np"
        assert b["tokens_np"][0]["pos"].dtype == np.float64 and b["tokens_np"][0]["valid"].shape == (K,)
        dr1, dr2, dr3 = (PTokData("val", seed=s_, token_drop=0.5, dir=d) for s_ in (1, 1, 2))
        o1, o2, o3 = dr1[0], dr2[0], dr3[0]
        assert np.array_equal(o1["valid"], o2["valid"]) and not np.array_equal(o1["valid"], o3["valid"]), "the drop is seeded by (seed, frame token)"
        gone = ~o1["valid"]
        assert gone.sum() > 5 and (o1["ytype"][gone] == 0).all() and not o1["y"][gone].any() and not o1["y"][:, gone].any() and (o1["ytype"][~gone] == arr["gt_type"][0][~gone]).all()
        assert PTokData("val", token_drop=0.0, dir=d)[0]["valid"].all()


def selftest():
    K = 6'''),
]

TEST = [
    ('''MUTANTS = [''', '''PTOK = os.path.join(ROOT, "training", "data", "lane_ptok_data.py")
MUTANTS = ['''),
]

if __name__ == "__main__":
    patch("lane_tokcache.py", TOKCACHE)
    patch("lane_ptok_data.py", PTOK)
    print("patched the work copies")
