"""E261 step 2 (loop 22, throw-away): out-of-fold choice of the operating point (tau, fork pruning) of rc2 (arm a) and of rc1 (the threshold-only
control t) on 20 seeded 11 / 11 VAL scene splits; card: experiments/E261_recall-weight-6-operating-point/experiment.md section 2.
Every number it prints is "VAL, selection-CV, layer 3" (grid and guards were fixed after the full-VAL frontier was seen).

    tools/env/_py14.sh temp/loop22/e261_cv.py           # the run E261.cv (through run.py)
    tools/env/_py14.sh temp/loop22/e261_cv.py --check   # layer 1 only: the KNOWN lines + the scene-sum path against card_any; writes nothing

Each (model, tau, fork pruning) graph set is assembled by evaluate.graphs_of (asm of the registry entry `recall`, cached raw outputs) and scored
ONCE on all 873 frames: bootstrap.frame_records, topo_metrics.frame_record and the per-piece DP pair bootstrap._vstats(g, False, 0.3); the frame
counts are added per scene. Every metric used here is a ratio of sums of those counts (graph_eval.aggregate, topo_metrics.pooled, vert100,
vert100_dp), so the metrics of a half are the pooled metrics of its 11 scene sums, equal to card_any on its frames (--check asserts it on all
frames). Not ratios of sums and not used: verts_per_frame, n_samples, split*/merge*_n_gt / _n_pred (per-frame means).
The scene sums of every point go to ~/exp_gcar_b123/loop22/e261_records.pkl after each point; a restart skips the points already there (delete
the file to re-score). The summary goes to experiments/E261_recall-weight-6-operating-point/e261_cv.json.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "2"  # shared box: 2 threads (set before numpy / numba load)
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only: the raw-output cache is required, the GPU belongs to the training

import json  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from functools import reduce  # noqa: E402

import numpy as np  # noqa: E402

import bootstrap as SB  # noqa: E402
import evaluate as EV  # noqa: E402
import sparse_metrics as SM  # noqa: E402
import topo_metrics as TM  # noqa: E402

RECORDS = os.path.expanduser("~/exp_gcar_b123/loop22/e261_records.pkl")
OUT = f"{EV.R}/experiments/E261_recall-weight-6-operating-point/e261_cv.json"
MODELS = {"rc2": "e18_D7_rc2", "rc1": "e123_D7_r3_rc1", "f1eq": "e18_D7_f1eq", "rc4": "e18_D7_rc4"}
TAUS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
PRUNES = (0.4, 0.5, 0.6)
GRID = [(m, t, p) for m in ("rc2", "rc1") for t in TAUS for p in PRUNES] + [("f1eq", 0.7, 0.4), ("rc4", 0.7, 0.4)]  # f1eq / rc4: report lines
REF = ("rc1", 0.7, 0.4)  # c = `recall` at its registered point
Q_MIN, DP_MAX, FL_MAX, DUP_STEP = 0.735, SM.DP_ORACLE, 0.062, 0.030  # the guards of the card (section 2)
DP_HALVES_MAX = 5  # referee: a point whose DP exceeds 14.3 in more than 5 of the 20 held-out halves is inadmissible
SEEDS = range(20)
DIFFS = ("lanes_found", "R@1_far", "Q", "topo_F1", "false_len_frac", "dup_frac")
SHOW = ("Q", "lanes_found", "R@1_far", "P@1_far", "topo_F1", "false_len_frac", "dup_frac", "vert100", "dp")
KNOWN = {  # layer 1, (value, decimals): the stored recall card, the E261.g line, the card's frontier table (rc2)
    ("rc1", 0.7, 0.4): {"Q": (0.7404, 4), "lanes_found": (0.7673, 4), "R@1_far": (0.7714, 4), "dp": (12.82, 2)},
    ("rc1", 0.2, 0.4): {"Q": (0.7355, 4), "lanes_found": (0.7778, 4), "R@1_far": (0.7811, 4), "dp": (13.48, 2)},
    ("rc2", 0.8, 0.4): {"Q": (0.7367, 4), "lanes_found": (0.7902, 4), "R@1_far": (0.7867, 4), "dp": (14.19, 2)},
    ("rc2", 0.7, 0.4): {"Q": (0.7351, 4), "lanes_found": (0.7929, 4), "R@1_far": (0.7900, 4), "dp": (14.35, 2)},
}


def _add(a, b):
    """field-wise sum of two count records (dicts by key, a key missing on one side is taken from the other; tuples element-wise)"""
    if isinstance(a, dict):
        return {k: _add(a[k], b[k]) if k in a and k in b else a.get(k, b.get(k)) for k in a.keys() | b.keys()}
    if isinstance(a, tuple):
        return tuple(x + y for x, y in zip(a, b))
    return a + b


def score(key, reg):
    """frame count records of one grid point (assembled by evaluate.graphs_of, scored once on all frames) + graphs, ground truths, info"""
    m, t, p = key
    e = dict(reg["recall"], run=MODELS[m], tau=t)
    g, gts, info = EV.graphs_of(f"{m}_{t}_{p}", e, "val", {"fork_prune": p}, t, require_cache=True)
    rec = SB.frame_records(g, gts)
    for r, pr, gt in zip(rec, g, gts):
        r.update(TM.frame_record(pr, gt))
        r["dnv"], r["dlen"] = SB._vstats(pr, False, 0.3)  # per-piece DP 0.3 m (the rule of vert100_dp)
    return rec, g, gts, info


def scene_sums(rec, members):
    return [reduce(_add, [rec[i] for i in ix]) for ix in members]


def metrics(srec, idx):
    """pooled metrics of the scene records idx (bootstrap.pooled + topo_metrics.pooled + per-piece DP per 100 m)"""
    sel = [srec[i] for i in idx]
    m = SB.pooled(sel)
    m.update(TM.pooled(sel))
    m["dp"] = 100.0 * sum(x["dnv"] for x in sel) / max(sum(x["dlen"] for x in sel), 1e-9)
    return m


def admissible(m, dup_ref):
    return m["Q"] >= Q_MIN and m["dp"] <= DP_MAX and m["false_len_frac"] <= FL_MAX and m["dup_frac"] <= dup_ref + DUP_STEP


def choose(mets, model, exclude=()):
    """the admissible point of the model with the highest lanes found (ties: higher Q, then higher tau); dup guard against REF on the same frames"""
    ok = [k for k in mets if k[0] == model and k not in exclude and admissible(mets[k], mets[REF]["dup_frac"])]
    return max(ok, key=lambda k: (mets[k]["lanes_found"], mets[k]["Q"], k[1]), default=None)


def layer1(key, m):
    bad = [f"{k} {m[k]:.{d}f} != {v:.{d}f}" for k, (v, d) in KNOWN[key].items() if abs(m[k] - v) > 0.5 * 10 ** -d + 1e-12]
    print(f"layer 1 {pt(key)}: {'ok' if not bad else 'FAILED ' + ', '.join(bad)}", flush=True)
    if bad:
        sys.exit(1)


def pt(k):
    return "none" if k is None else f"{k[0]} {k[1]}/{k[2]}"


def line(k, m):
    return f"{pt(k):16s} " + " ".join(f"{x} {m[x]:.4f}" if x != "dp" else f"dp {m[x]:.2f}" for x in SHOW)


def save(store):
    os.makedirs(os.path.dirname(RECORDS), exist_ok=True)
    tmp = f"{RECORDS}.tmp{os.getpid()}"  # written whole, then renamed
    with open(tmp, "wb") as f:
        pickle.dump(store, f)
    os.replace(tmp, RECORDS)


def collect(check, reg, ids, members):
    """scene records of every point (KNOWN first, each checked as soon as it is scored); --check: KNOWN only, nothing read or written"""
    store = {"points": {}} if check or not os.path.exists(RECORDS) else pickle.load(open(RECORDS, "rb"))
    assert store.get("scenes", list(ids)) == list(ids), "the records were made on other scenes"
    store["scenes"] = list(ids)
    keys = list(KNOWN) + ([] if check else [k for k in GRID if k not in KNOWN])
    for key in keys:
        if key not in store["points"]:
            t0 = time.time()
            rec, g, gts, info = score(key, reg)
            sha = EV.digest(gts)
            assert store.setdefault("gt_sha", sha) == sha and len(gts) == sum(len(ix) for ix in members), "frames or ground truth differ"
            store["points"][key] = scene_sums(rec, members)
            store.setdefault("cache", {})[key[0]] = info["cache"]
            if check and key == REF:  # the scene-sum path equals card_any (+ vert100_dp) on the same graphs
                ref = SM.card_any(g, gts, extra=True)
                ref["dp"] = SM.vert100_dp(g, 0.3)
                got = metrics(store["points"][key], range(len(members)))
                bad = [k for k in SHOW + ("S", "F1@1.0", "J") if abs(got[k] - ref[k]) > 1e-9]
                print("scene sums == card_any on all frames:", "ok" if not bad else f"FAILED {bad}", flush=True)
                if bad:
                    sys.exit(1)
            del rec, g, gts
            if not check:
                save(store)
            print(f"scored {pt(key)} in {time.time() - t0:.0f} s ({info['cache']})", flush=True)
        if key in KNOWN:
            layer1(key, metrics(store["points"][key], range(len(members))))
    return store


def stats(v):
    v = np.array([x for x in v if x is not None], float)
    return dict(n=len(v), mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else float("nan"), min=float(v.min()), max=float(v.max())) if len(v) else dict(n=0)


def main():
    check = "--check" in sys.argv[1:]
    reg = EV.registry()
    sc = EV.scenes("val")
    ids = np.unique(sc)
    members = [np.nonzero(sc == s)[0] for s in ids]
    assert len(ids) == 22, f"{len(ids)} VAL scenes, the card expects 22"
    store = collect(check, reg, ids, members)
    if check:
        return
    P = store["points"]
    print(f"\nE261.cv - VAL {len(sc)} frames, 22 scenes; VAL, selection-CV, layer 3; raw caches: {store['cache']}")
    ALL = range(len(ids))
    mall = {k: metrics(P[k], ALL) for k in GRID}

    rows, dp_fail = [], Counter()
    print("\nper split (choice on half A, read on half B; a = rc2, t = rc1, c = recall rc1 0.7/0.4):")
    for seed in SEEDS:
        inA = set(np.random.default_rng(seed).choice(ids, 11, replace=False).tolist())
        A = [j for j, s in enumerate(ids) if s in inA]
        B = [j for j, s in enumerate(ids) if s not in inA]
        mA = {k: metrics(P[k], A) for k in GRID}
        mB = {k: metrics(P[k], B) for k in GRID}
        dp_fail.update(k for k in GRID if mB[k]["dp"] > DP_MAX)
        a, t = choose(mA, "rc2"), choose(mA, "rc1")
        row = dict(seed=seed, a=a, t=t, dp_a=mB[a]["dp"] if a else None, adm_a_B=admissible(mB[a], mB[REF]["dup_frac"]) if a else None)
        for nm, x, y in (("a-c", a, REF), ("a-t", a, t), ("t-c", t, REF)):
            row[nm] = {d: (mB[x][d] - mB[y][d] if x and y else None) for d in DIFFS}
        rows.append(row)
        print(f"seed {seed:2d} a {pt(a)} t {pt(t)}  " + "  ".join(f"{nm}: LF {row[nm]['lanes_found']:+.4f} R@1far {row[nm]['R@1_far']:+.4f} Q {row[nm]['Q']:+.4f}"
              if row[nm]["Q"] is not None else f"{nm}: -" for nm in ("a-c", "a-t", "t-c")) + (f"  DP(a) {row['dp_a']:.2f}" if a else ""), flush=True)

    summary = {nm: {d: stats([r[nm][d] for r in rows]) for d in DIFFS} for nm in ("a-c", "a-t", "t-c")}
    summary["dp_a_B"] = stats([r["dp_a"] for r in rows])
    n_dp = sum(1 for r in rows if r["dp_a"] is not None and r["dp_a"] > DP_MAX)
    n_inadm = sum(1 for r in rows if r["adm_a_B"] is False)
    print("\nheld-out halves (20 splits): mean / SD / min / max")
    for nm in ("a-c", "a-t", "t-c"):
        for d in DIFFS:
            s = summary[nm][d]
            print(f"  {nm} {d:15s} " + (f"{s['mean']:+.4f} / {s['sd']:.4f} / {s['min']:+.4f} / {s['max']:+.4f} (n {s['n']})" if s["n"] else "no split"))
    s = summary["dp_a_B"]
    if s["n"]:
        print(f"  DP(a) on B        {s['mean']:.2f} / {s['sd']:.2f} / {s['min']:.2f} / {s['max']:.2f}; > {DP_MAX} in {n_dp} of {len(rows)} halves; a inadmissible on B (any guard) in {n_inadm}")
    hist = {m: Counter(pt(r[m]) for r in rows) for m in ("a", "t")}
    print("  chosen points:", {m: dict(h.most_common()) for m, h in hist.items()})

    print("\nall frames (873; in-sample), every point; adm = guards on all frames, dpB = held-out halves with DP > 14.3:")
    for k in GRID:
        print(f"  {line(k, mall[k])} adm {int(admissible(mall[k], mall[REF]['dup_frac']))} dpB {dp_fail[k]}")
    unstable = {k for k in GRID if dp_fail[k] > DP_HALVES_MAX}
    choice = {m: choose(mall, m, unstable) for m in ("rc2", "rc1")}
    plain = {m: choose(mall, m) for m in ("rc2", "rc1")}
    print("\nchoice on all 22 scenes (in-sample, VAL, selection-CV, layer 3; points with DP > 14.3 in more than 5 held-out halves excluded):")
    for m, name in (("rc2", "recall6"), ("rc1", "recall_thr")):
        k = choice[m]
        print(f"  registry candidate {name}: " + (f"{line(k, mall[k])} dpB {dp_fail[k]}" if k else "no admissible point")
              + (f"  (without the DP-halves rule: {pt(plain[m])})" if plain[m] != k else ""))
    a, t = choice["rc2"], choice["rc1"]
    for nm, x, y in (("a-c", a, REF), ("a-t", a, t), ("t-c", t, REF)):
        if x and y:
            print(f"  {nm} all frames: " + "  ".join(f"{d} {mall[x][d] - mall[y][d]:+.4f}" for d in DIFFS))

    head = {f"{nm} {d}": summary[nm][d].get("mean") for nm in ("a-c", "a-t") for d in ("lanes_found", "R@1_far", "Q")}  # first: run.py --result keeps 400 characters
    res = dict(label="VAL, selection-CV, layer 3", recall6=pt(a), recall_thr=pt(t), held_out_mean=head, dp_a_over_B=n_dp, a_inadmissible_B=n_inadm,
               frames=len(sc), scenes=len(ids), caches=store["cache"], guards=dict(Q_min=Q_MIN, dp_max=DP_MAX, fl_max=FL_MAX, dup_step=DUP_STEP, dp_halves_max=DP_HALVES_MAX),
               held_out=summary, chosen=hist, splits=[dict(r, a=pt(r["a"]), t=pt(r["t"])) for r in rows],
               all_frames={pt(k): {x: mall[k][x] for x in SHOW} | dict(dp_halves_over=dp_fail[k]) for k in GRID})
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, default=float)
    print("summary:", OUT)


if __name__ == "__main__":
    main()
