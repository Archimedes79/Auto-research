import os
R = os.environ["BFLC_ROOT"]
p = os.path.join(R, "training/grid/test_lane_map.py")
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


# ---------------------------------------------------------------- docstring
sub("""A mutant = one textual change of the source (the pose error applied with the transposed rotation,""", """`potential` is also compared, in the unmutated run, with an INDEPENDENT re-implementation of the card's step 0 (`reference_potential`: the scratch script step0_map2.py with the cut of the in-intersection connector ranges written out again, loop by loop) to 1e-12.

A mutant = one textual change of the source (the cut of the in-intersection connector ranges skipped, the pose error applied with the transposed rotation,""")

# ---------------------------------------------------------------- mutants
sub('''    ("the pose error is applied with the transposed rotation",''', '''    ("the cut of the in-intersection connector ranges is skipped (the real-frame test)", '            keep[sg["i0"]: sg["i1"] + 1] = False', "            pass", "fixture"),
    ("the cut is skipped", '            keep[sg["i0"]: sg["i1"] + 1] = False', "            pass"),
    ("the frame's points bypass the cut", "return frame_pts(self.pieces(i))", 'return [np.asarray(pc["pts"], float) for pc in self.pieces(i)]', "fixture"),
    ("a connector range outside an intersection is cut", 'if sg["kind"] == "connector" and sg["in_intersection"]:\\n            keep[', 'if sg["kind"] == "connector":\\n            keep['),
    ("a lane range in an intersection is cut", 'if sg["kind"] == "connector" and sg["in_intersection"]:\\n            keep[', 'if sg["in_intersection"]:\\n            keep['),
    ("the end of a connector range is not cut", 'keep[sg["i0"]: sg["i1"] + 1] = False', 'keep[sg["i0"]: sg["i1"]] = False'),
    ("a run of one point is kept", "if b - a >= 2]", "if b - a >= 1]"),
    ("the counts are rounded down", "np.rint(np.concatenate(C))", "np.floor(np.concatenate(C))"),
    ("the counts are rounded up", "np.rint(np.concatenate(C))", "np.ceil(np.concatenate(C))"),
    ("mode other takes a fixed frame", "            src = int(rng.randint(len(self.tokens)))\\n            while self.scene[src] == self.scene[i]:\\n                src = int(rng.randint(len(self.tokens)))", "            src = int(self.plan[i])"),
    ("mode other may take a frame of the same scene", "            while self.scene[src] == self.scene[i]:", "            while False:"),
    ("mode other has the stream of mode train", "+ ([1] if other else [])", "+ []"),
    ("mode other has no counters", 'if mode in ("train", "other") else None       # the draw', 'if mode == "train" else None       # the draw'),
    ("mode S without a plan fails without a message", 'assert self.plan is not None, "mode S needs the scene plan', 'assert True or "mode S needs the scene plan'),
    ("the TRAIN guard ignores the TEST frames", 'for r in list(splits["val"]) + list(splits["test"])}', 'for r in list(splits["val"])}'),
    ("the TRAIN guard lets a VAL frame through", "assert not both,", "assert True or not both,"),
    ("the TRAIN guard does not check the rows", 'assert len(set(rows)) == len(rows) and sorted(int(r) for r in rows) == sorted(int(r) for r in splits["train"]),', "assert True or len(set(rows)) == len(rows),"),
    ("the TRAIN guard does not check the tokens", 'assert all(train_tokens[int(r)] == t for r, t in zip(rows, tokens)),', "assert True or all(train_tokens[int(r)] == t for r, t in zip(rows, tokens)),"),
    ("the TRAIN guard is not run by check", 'res["train_guard"] = train_guard(ds.tok.rows, ds.tokens, splits, *tok_files)', 'res["train_guard"] = dict(train_frames=0, val_test_frames=0)', "check"),
    ("the pose error is applied with the transposed rotation",''')
sub('''def get(r, path):''', '''def reference_potential(n):
    """an INDEPENDENT re-implementation of the card's step 0 on the cut map (the scratch script step0_map2.py of the card, with the cut of the in-intersection connector ranges written out again point by point; it shares nothing with `lane_map_data` but `sd_lib`, the GT store and the cache readers)
    for n evenly spaced VAL frames of B' seed 0 -> the numbers of `potential`'s JSON (classes, by_bin, samples_per_frame)"""
    import hashlib
    from scipy.spatial import cKDTree
    for sub_ in ("training/data", "training/data/prep", "training/grid", ".."):
        sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
    import lane_gt_data as LD
    import lane_ptok_data as LP
    import lane_tok_data as LT
    import lane_tokcache as LC
    import sd_lib as SD
    c = LC.TokCache(LP.cache_dir("val", 0, "e292_a"))
    ds = LT.TokData("val", res=2.0, feats=False)
    lane_gt, _ = LD.lane_store()

    def cut_points(piece):
        pts, drop, runs, cur = np.asarray(piece["pts"], float), set(), [], []
        for s_ in piece["segments"]:
            if s_["kind"] == "connector" and s_["in_intersection"]:
                drop.update(range(s_["i0"], s_["i1"] + 1))
        for k in range(len(pts)):
            if k in drop:
                if len(cur) >= 2:
                    runs.append(np.array(cur))
                cur = []
            else:
                cur.append(pts[k])
        if len(cur) >= 2:
            runs.append(np.array(cur))
        return runs

    def samples(g, rng, degrade, shift=None, yaw=0.0):
        sd = SD.build_sd(dict(pts=[r for pc in g["pieces"] for r in cut_points(pc)]), [])
        if degrade:
            sd = SD.degrade_sd(sd, rng, drop=0.1)
        P, T = [], []
        for poly, cnt in sd["roads"]:
            poly = np.asarray(poly, float)
            if len(poly) < 2:
                continue
            if degrade:
                cy, sy = np.cos(yaw), np.sin(yaw)
                poly = poly @ np.array([[cy, -sy], [sy, cy]]).T + shift
            seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
            s_ = np.concatenate([[0], np.cumsum(seg)])
            if s_[-1] < 1e-6:
                continue
            t = np.linspace(0, s_[-1], max(int(np.ceil(s_[-1] / 2.0)), 1) + 1)
            q = np.stack([np.interp(t, s_, poly[:, 0]), np.interp(t, s_, poly[:, 1])], 1)
            tg = np.gradient(q, axis=0)
            tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-9)
            P.append(q)
            T.append(tg)
        return (np.concatenate(P), np.concatenate(T)) if P else (np.zeros((0, 2)), np.zeros((0, 2)))

    def maps_of(i, degrade):
        g = lane_gt.load(c.tokens[i])
        rng = np.random.RandomState(int(hashlib.md5(c.tokens[i].encode()).hexdigest()[:8], 16))
        ang = rng.uniform(0, 2 * np.pi)
        sh = rng.uniform(1, 2) * np.array([np.cos(ang), np.sin(ang)])
        return samples(g, rng, degrade, sh, np.deg2rad(rng.randn()))

    N = len(c)
    cls = ("novel", "dup", "false", "deliv_true", "deliv_false")
    res = {(mode, k): [] for mode in ("exact", "error", "null") for k in cls}
    bins, nsd = [], []
    for i in np.linspace(0, N - 1, n).astype(int):
        t = ds.table(int(i))
        gpos, gdir = t["pos"], t["dirv"]
        if len(gpos) == 0:
            continue
        pos, dirv, valid, ep = (np.array(c.f[k][i], float) if k != "valid" else np.array(c.f[k][i]) for k in ("pos", "dirv", "valid", "e_p"))
        deliv, weak = valid & (ep >= 0.5), valid & ~(valid & (ep >= 0.5))
        near = (np.linalg.norm(pos[:, None] - gpos[None], axis=2) <= 1.5) & (dirv @ gdir.T >= 0.5)
        miss = ~((near & deliv[:, None]).any(0))
        sel = dict(novel=weak & (near & miss[None]).any(1), dup=weak & near.any(1) & ~(near & miss[None]).any(1), false=weak & ~near.any(1), deliv_true=deliv & near.any(1), deliv_false=deliv & ~near.any(1))
        maps = dict(exact=maps_of(i, False), error=maps_of(i, True), null=maps_of((i + 97) % N, True))
        nsd.append(len(maps["exact"][0]))
        for mode, (P, T) in maps.items():
            if len(P) == 0:
                sup, dist = np.zeros(len(pos), bool), np.full(len(pos), 99.0)
            else:
                dist, jj = cKDTree(P).query(pos)
                sup = (dist <= 3.0) & (np.abs((dirv * T[jj]).sum(1)) >= 0.8)
            for k in cls:
                res[(mode, k)].append(np.stack([dist[sel[k]], sup[sel[k]]], 1))
            if mode == "error":
                bins.append(np.stack([ep[weak], sup[weak], near.any(1)[weak], (near & miss[None]).any(1)[weak]], 1))
    out = dict(samples_per_frame=float(np.mean(nsd)), classes={}, by_bin=[])
    for k in cls:
        a = {m: np.concatenate(res[(m, k)]) for m in ("exact", "error", "null")}
        out["classes"][k] = dict(n=len(a["exact"]), supported_exact=float(a["exact"][:, 1].mean()), supported_error=float(a["error"][:, 1].mean()), supported_other_scene=float(a["null"][:, 1].mean()), median_dist_exact=float(np.median(a["exact"][:, 0])), median_dist_error=float(np.median(a["error"][:, 0])))
    B = np.concatenate(bins)
    for lo, hi in ((0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)):
        m = (B[:, 0] >= lo) & (B[:, 0] < hi)
        t_, f_, s_, u_ = m & (B[:, 2] > 0.5), m & (B[:, 2] < 0.5), m & (B[:, 1] > 0.5), m & (B[:, 1] < 0.5)
        out["by_bin"].append(dict(n=int(m.sum()), true_share=float(t_.sum() / m.sum()), supported_if_true=float(B[t_, 1].mean()), supported_if_false=float(B[f_, 1].mean()), novel_share_supported=float(B[s_, 3].mean()), novel_share_unsupported=float(B[u_, 3].mean())))
    return out


def compare_reference(path, n=60):
    """`potential --n 60`'s JSON against the independent re-implementation: every number to 1e-12"""
    r, ref = json.load(open(path)), reference_potential(n)
    pairs = [(r["samples_per_frame"], ref["samples_per_frame"], "samples_per_frame")]
    pairs += [(r["classes"][k][kk], vv, f"{k}.{kk}") for k, v in ref["classes"].items() for kk, vv in v.items()]
    pairs += [(r["by_bin"][i][kk], vv, f"bin{i}.{kk}") for i, b in enumerate(ref["by_bin"]) for kk, vv in b.items()]
    bad = [(t, a, b) for a, b, t in pairs if abs(a - b) > 1e-12]
    assert not bad, f"potential differs from its independent re-implementation: {bad[:3]}"
    return len(pairs)


def get(r, path):''')
sub('''    for p, want in PINS[kind].items():
        got = get(r, p)
        assert abs(got - want)''', '''    for p, want in PINS[kind].items():
        try:
            got = get(r, p)
        except (KeyError, IndexError, TypeError) as e:
            raise AssertionError(f"{kind} {p}: not in the result ({e!r})")
        assert abs(got - want)''')
sub('''            print(f"{kind_} pins ok", flush=True)
''', '''            print(f"{kind_} pins ok", flush=True)
        print(f"potential = its independent re-implementation on the cut map: {compare_reference(os.path.join(tmp_, 'potential.json'))} numbers to 1e-12", flush=True)
''')
sub('''import concurrent.futures as cf
import json''', '''import concurrent.futures as cf
import json''')
sub('''import tempfile

ROOT =''', '''import tempfile

import numpy as np

ROOT =''')

open(p, "w", encoding="utf-8").write(s)
print("patched tl mutants")
