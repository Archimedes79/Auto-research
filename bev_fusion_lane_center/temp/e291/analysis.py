"""E291 analysis (CPU, scratch of the card): where does a read-out gain, and does the gate know when the count is wrong?

    wsl bash tools/env/_py14.sh analysis.py splits <run_a> <card_a> <run_b> <card_b> [<run_a2> <card_a2> <run_b2> <card_b2> ...] [--n 1000]
        the gain A - B of the topo F1 (one group of four words per seed; the counts files `runs/<run>/<card stem>.counts.pkl` of `jends_eval.py score`) split by the junction GT of the frame: frames with an arm of at least two lanes in a direction
        (`multi`) against frames whose arms have one lane or none (`single`) against frames without junction GT; by the range of the nearest arm centre (< 25 m, 25 - 40 m, >= 40 m); and the cross of both. Per class: frames, topo F1 of A and B, the gain per seed,
        the pooled mean over the seeds with the paired scene bootstrap interval (`jdiag.pooled_boot`: one scene draw per resample for all seeds, the scenes of the class only; a class with fewer than 4 scenes has no interval).
    wsl bash tools/env/_py14.sh analysis.py gate <run_read_noisy> <run_read_clean>
        the prior numbers of two read-outs of ONE model (`jends_eval.py dump --prior-mode`: the keypoints are the same in every read-out): per keypoint that saw a count in both, is the count of the noisy read-out the count of the clean one, and how large was the
        term ||sigmoid(g) f|| against the token ||t||: mean ratio for right and wrong counts, and their correlation.
    wsl bash tools/env/_py14.sh analysis.py selftest         the split on the stored E312 counts (`runs/e312_p_s0/ends_card_snap_thr0.1.counts.pkl`: the snapped graph against the plain graph of the same file) and toy classes / dumps with known answers
"""
import os
import pickle
import sys

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.getcwd()
for sub in ("evaluation", "training", "training/grid", "training/data", "tools/viz", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))

BANDS = ((0.0, 25.0, "< 25 m"), (25.0, 40.0, "25-40 m"), (40.0, 1e9, ">= 40 m"))


def frame_class(jg):
    """(multi, range band index) of a frame from its junction GT; (None, None) without junction GT. multi: some arm has two or more entry or exit ends (lanes); range: the distance of the nearest arm centre from the ego at the origin"""
    if jg is None or not len(jg["arms"]):
        return None, None
    multi = any(max(int(a["n_in"]), int(a["n_out"])) >= 2 for a in jg["arms"])
    near = min(float(np.linalg.norm(np.asarray(a["c"], float))) for a in jg["arms"])
    return multi, next(i for i, (lo, hi, _) in enumerate(BANDS) if lo <= near < hi)


def groupings(cls):
    """{grouping: {class name: frame indices}} from the list of (multi, band) per frame"""
    names = lambda m, b: "no junction GT" if m is None else f"{'multi' if m else 'single'}"  # noqa: E731
    g = {"arms": {}, "range": {}, "arms x range": {}}
    for i, (m, b) in enumerate(cls):
        g["arms"].setdefault(names(m, b), []).append(i)
        g["range"].setdefault("no junction GT" if m is None else BANDS[b][2], []).append(i)
        g["arms x range"].setdefault("no junction GT" if m is None else f"{names(m, b)}, {BANDS[b][2]}", []).append(i)
    return g


def restrict(data, idx):
    idx = list(idx)
    return [[dict(scenes=[x["scenes"][i] for i in idx], head=[x["head"][i] for i in idx]) for x in pair] for pair in data]


def split_gain(data, cls, n=1000, seed=0, numbers=None, min_scenes=4):
    """data: one [A, B] pair of loaded counts per seed (A, B: dict scenes, head); cls: (multi, band) per frame -> {grouping: [(class, frames, scenes, topo F1 A, topo F1 B, gain per seed, pooled mean, (lo, hi) or None)]}"""
    import jdiag
    if numbers is None:
        import qgraph_eval as QE
        numbers = QE.graph_numbers
    out = {}
    for name, groups in groupings(cls).items():
        rows = []
        for c, idx in sorted(groups.items()):
            d = restrict(data, idx)
            sc = len(set(d[0][0]["scenes"]))
            fa = float(np.mean([numbers(p[0]["head"])["topo_F1"] for p in d]))
            fb = float(np.mean([numbers(p[1]["head"])["topo_F1"] for p in d]))
            r = jdiag.pooled_boot(d, n=n, seed=seed, numbers=numbers) if sc >= min_scenes else None
            per = [numbers(p[0]["head"])["topo_F1"] - numbers(p[1]["head"])["topo_F1"] for p in d]
            rows.append((c, len(idx), sc, fa, fb, per, float(np.mean(per)), None if r is None else r["ci95"]))
        out[name] = rows
    return out


def show(res):
    for name, rows in res.items():
        print(f"\n{name}\n  {'class':22s} frames scenes  topo F1 A   topo F1 B   gain per seed          pooled   95 % interval of the pooled gain")
        for c, nf, sc, fa, fb, per, mean, ci in rows:
            print(f"  {c:22s} {nf:6d} {sc:6d}  {fa:9.4f}  {fb:10.4f}   {' '.join(f'{p:+.4f}' for p in per):22s} {mean:+.4f}   " + ("[%+.4f, %+.4f]" % ci if ci else "(fewer than 4 scenes)"))


def frame_classes():
    """(multi, band) of every VAL frame in the order of the counts files (`qgraph_eval.val_frames`), from the lane GT"""
    import qgraph_eval as QE
    import qgraph_view as QV
    import jgt as JG
    return [frame_class(JG.junction_gt(QV.load_gt(f.token))) for f in QE.val_frames()]


def gate(run_noisy, run_clean):
    """per keypoint that saw a count in both read-outs: right (the same count as the clean read-out) or wrong, and the size of the term against the token in the NOISY read-out"""
    load = lambda r: pickle.load(open(os.path.join(ROOT, "runs", r, "ends_val.pkl"), "rb"))  # noqa: E731
    a, b = load(run_noisy), load(run_clean)
    assert [f["token"] for f in a] == [f["token"] for f in b], "the two read-outs must be dumps of the same frames"
    ratio, right = [], []
    for fa, fb in zip(a, b):
        assert np.array_equal(fa["kp_pos"], fb["kp_pos"]), "the keypoints of one model are the same in every read-out: these dumps are of two models"
        m = fa["valid"].reshape(-1) & (fa["prior"]["n"] >= 0) & (fb["prior"]["n"] >= 0)
        ratio.append((fa["prior"]["term"] / np.maximum(fa["prior"]["tok"], 1e-9))[m])
        right.append((fa["prior"]["n"] == fb["prior"]["n"])[m])
    ratio, right = np.concatenate(ratio), np.concatenate(right)
    res = dict(n=int(len(ratio)), right=int(right.sum()), wrong=int((~right).sum()), ratio_right=float(ratio[right].mean()) if right.any() else float("nan"), ratio_wrong=float(ratio[~right].mean()) if (~right).any() else float("nan"),
               corr=float(np.corrcoef(ratio, right.astype(float))[0, 1]) if right.any() and (~right).any() else float("nan"))
    print(f"{run_noisy} against {run_clean}: {res['n']} keypoints with a count in both, {res['right']} right, {res['wrong']} wrong; term / token: right {res['ratio_right']:.4f}, wrong {res['ratio_wrong']:.4f}; correlation with `right`: {res['corr']:+.3f}")
    return res


def selftest():
    import qgraph_eval as QE
    # classes: the arms of a toy junction
    arm = lambda c, a, b: dict(c=np.array(c, float), t=np.array([0.0, 1.0]), n_in=a, n_out=b, zone=0)  # noqa: E731
    assert frame_class(None) == (None, None) and frame_class(dict(arms=[])) == (None, None)
    assert frame_class(dict(arms=[arm((0, 30), 1, 1), arm((0, -50), 1, 0)])) == (False, 1) and frame_class(dict(arms=[arm((0, 20), 1, 1), arm((0, 60), 3, 1)])) == (True, 0) and frame_class(dict(arms=[arm((60, 0), 1, 2)])) == (True, 2)
    g = groupings([(None, None), (False, 0), (True, 0), (True, 2), (True, 2)])
    assert g["arms"] == {"no junction GT": [0], "single": [1], "multi": [2, 3, 4]} and g["range"][">= 40 m"] == [3, 4] and g["arms x range"]["multi, >= 40 m"] == [3, 4]
    # toy counts: the gain of A over B is +0.1 on the frames of the first four scenes and 0 elsewhere (the metric of a set of frames = their mean of `f`)
    toy = lambda cs: {"topo_F1": float(np.mean([c["f"] for c in cs]))}  # noqa: E731
    scenes = [s for s in range(8) for _ in range(3)]
    A = dict(scenes=scenes, head=[dict(f=0.6 if s < 4 else 0.5) for s in scenes])
    B = dict(scenes=scenes, head=[dict(f=0.5) for _ in scenes])
    cls = [(True, 0) if s < 4 else (False, 1) for s in scenes]
    res = split_gain([[A, B]] * 2, cls, n=100, numbers=toy)
    rows = {c: r for c, *r in res["arms"]}
    assert abs(rows["multi"][4][0] - 0.1) < 1e-9 and abs(rows["single"][4][0]) < 1e-9 and rows["multi"][0] == 12 and rows["multi"][1] == 4, rows
    assert rows["multi"][6][0] > 0.0999 and rows["single"][6][1] < 1e-9, rows
    # the stored E312 counts: the snapped graph (head) against the plain graph (base) of the same file, VAL 873 frames; every frame in exactly one class of every grouping
    p = os.path.join(ROOT, "runs", "e312_p_s0", "ends_card_snap_thr0.1.counts.pkl")
    C = pickle.load(open(p, "rb"))
    n = len(C["head"])
    cls = [(i % 3 == 0, i % 3) if i % 7 else (None, None) for i in range(n)]
    d = [[dict(scenes=C["scenes"], head=C["head"]), dict(scenes=C["scenes"], head=C["base"])]]
    r = split_gain(d, cls, n=30)
    assert n == 873 and all(sum(x[1] for x in rows_) == n for rows_ in r.values())
    allrow = QE.graph_numbers(C["head"])["topo_F1"] - QE.graph_numbers(C["base"])["topo_F1"]
    whole = split_gain(d, [(True, 0)] * n, n=30)["arms"][0]
    assert abs(whole[5][0] - allrow) < 1e-12 and abs(allrow - 0.057) < 0.05, (whole, allrow)
    print(f"  stored E312 counts: snapped minus plain graph topo F1 {allrow:+.4f} on {n} frames, interval {whole[7][0]:+.4f} .. {whole[7][1]:+.4f}")
    # the gate numbers of two read-outs of one model: the keypoints and the counts of a toy dump with a known answer
    import tempfile
    tmp = tempfile.mkdtemp(prefix="e291_gate_")
    me = sys.modules[__name__]
    old, me.ROOT = ROOT, tmp
    try:
        rs = np.random.RandomState(0)
        K = 6
        mk = lambda n_, term: [dict(token=f"t{i}", kp_pos=np.arange(2 * K * 2, dtype=float).reshape(2 * K, 2) + i, valid=np.ones((2, K), bool), prior=dict(n=n_[i], term=term[i], tok=np.ones(2 * K))) for i in range(5)]  # noqa: E731
        clean_n = [rs.randint(0, 4, 2 * K).astype(float) for _ in range(5)]
        noisy_n = [c.copy() for c in clean_n]
        wrong = [rs.rand(2 * K) < 0.4 for _ in range(5)]
        for nn_, w in zip(noisy_n, wrong):
            nn_[w] += 1
        term = [np.where(w, 0.01, 0.2) for w in wrong]
        for name, ns in (("noisy", noisy_n), ("clean", clean_n)):
            os.makedirs(os.path.join(tmp, "runs", name))
            pickle.dump(mk(ns, term), open(os.path.join(tmp, "runs", name, "ends_val.pkl"), "wb"))
        g = gate("noisy", "clean")
        assert g["n"] == 60 and g["wrong"] == int(sum(w.sum() for w in wrong)) and abs(g["ratio_right"] - 0.2) < 1e-9 and abs(g["ratio_wrong"] - 0.01) < 1e-9 and g["corr"] > 0.99, g
    finally:
        me.ROOT = old
    print("analysis selftest ok")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "selftest":
        selftest()
    elif cmd == "gate" and len(sys.argv) == 4:
        gate(sys.argv[2], sys.argv[3])
    elif cmd == "splits":
        args, n = sys.argv[2:], 1000
        if "--n" in args:
            n = int(args[args.index("--n") + 1])
            del args[args.index("--n"):args.index("--n") + 2]
        assert args and len(args) % 4 == 0, "splits: groups of four words (run A, card A, run B, card B), one per seed"
        data = [[pickle.load(open(os.path.join(ROOT, "runs", args[i + j], args[i + j + 1] + ".counts.pkl"), "rb")) for j in (0, 2)] for i in range(0, len(args), 4)]
        for pair in data:
            assert pair[0]["scenes"] == pair[1]["scenes"] == data[0][0]["scenes"], "equal scene lists"
        cls = frame_classes()
        assert len(cls) == len(data[0][0]["head"]), (len(cls), len(data[0][0]["head"]))
        show(split_gain(data, cls, n=n))
    else:
        print(__doc__)
        sys.exit(2)
