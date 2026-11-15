"""E354: the pair flip rate of a run at any threshold, and against a control at MATCHED delivery with a scene bootstrap. CPU, VAL, from the dumps runs/<run>/ends_val.pkl; the definitions are `jtime.py`'s (imported, not edited).

    wsl bash tools/env/_py14.sh jflip.py rate    --run e354_r_s0 --seed 0 [--thr 0.1]                          # prints; at 0.1 it is the flip_rate of `jtime.py ceiling --run`
    wsl bash tools/env/_py14.sh jflip.py compare --run e354_r_s0 --control e354_z_s0 --seed 0 [--thr 0.1]      # -> runs/<run>/flip_<control>.json
    wsl bash tools/env/_py14.sh jflip.py selftest

Flip rate (E290, `jtime.ceiling`): over the pairs of consecutive keyframes of a scene, the TRUE pairs of t-1 delivered (p >= thr) at t-1 or, through the counterpart keypoints, as a true pair at t; the share delivered in exactly one of
the two. A run that delivers fewer pairs flips less for free, so `compare` also reads the run at MATCHED delivery: its threshold is chosen so that its number of true pairs delivered at t-1 is the control's at `--thr` (the nearest
count the pair probabilities allow; both counts are printed). Interval: a paired scene bootstrap of the difference of the POOLED flip rates (run - control; the scenes are the units, the same draw for both, 2000 resamples, seed 0),
at the equal threshold and at matched delivery.
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jtime as JTM  # noqa: E402

N_BOOT, BOOT_SEED = 2000, 0


def collect(run, seed):
    """-> {scene id: (p0 (n,), q1 (n,))} over the true pairs of t-1 of every pair of consecutive keyframes: p0 = the pair probability at t-1, q1 = the probability of its counterpart pair at t when that is a true pair, else 0 (never delivered)"""
    run, ends, gt, frames, scenes, infos = JTM._setup(SimpleNamespace(run=run, seed=seed))
    out = {}
    for sid, (_, toks) in scenes.items():
        P0, Q1 = [], []
        for t0, t1 in zip(toks[:-1], toks[1:]):
            if t0 not in ends or t1 not in ends:
                continue
            e0, e1 = ends[t0], ends[t1]
            p0, p1 = JTM._prob(e0), JTM._prob(e1)
            T0, T1 = JTM._truth(e0, gt[frames[t0].row]), JTM._truth(e1, gt[frames[t1].row])
            cpi, cpo = JTM._associate(e0, e1, JTM.lane_frame(infos[t0]), JTM.lane_frame(infos[t1]))
            for i, j in zip(*np.nonzero(T0)):
                ci, cj = cpi[i], cpo[j]
                P0.append(float(p0[i, j]))
                Q1.append(float(p1[ci, cj]) if (ci >= 0 and cj >= 0 and T1[ci, cj]) else 0.0)
        out[int(sid)] = (np.asarray(P0), np.asarray(Q1))
    return out


def counts(data, thr):
    """-> {scene: (flips, either, delivered at t-1)} at the threshold"""
    return {s: (int(((p0 >= thr) != (q1 >= thr)).sum()), int(((p0 >= thr) | (q1 >= thr)).sum()), int((p0 >= thr).sum())) for s, (p0, q1) in data.items()}


def rate(data, thr):
    c = np.array(list(counts(data, thr).values())).reshape(-1, 3).sum(0)
    return dict(thr=float(thr), flips_one_only=int(c[0]), flips_total_either=int(c[1]), true_pairs_t0_delivered=int(c[2]), flip_rate=c[0] / max(c[1], 1))


def matched_thr(data, n):
    """the threshold at which the run delivers n true pairs at t-1: the n-th largest pair probability (ties and too few pairs give the nearest count: read `true_pairs_t0_delivered`)"""
    p = np.sort(np.concatenate([p0 for p0, _ in data.values()]))[::-1]
    p = p[p > 0]
    assert n > 0 and len(p), "no true pair delivered"
    return float(p[min(n, len(p)) - 1])


def boot_diff(ca, cb, n_boot=N_BOOT, seed=BOOT_SEED):
    """paired scene bootstrap of the pooled flip-rate difference a - b; ca, cb = `counts` of the two runs -> (difference, lo, hi) with the 95 % interval"""
    scenes = sorted(set(ca) & set(cb))
    assert scenes and len(scenes) == len(ca) == len(cb), "the two dumps must cover the same scenes"
    A, B = (np.array([[c[s][0], c[s][1]] for s in scenes], float) for c in (ca, cb))
    pooled = lambda M: M[..., 0].sum(-1) / np.maximum(M[..., 1].sum(-1), 1)  # noqa: E731
    ix = np.random.RandomState(seed).randint(0, len(scenes), (n_boot, len(scenes)))
    d = pooled(A[ix]) - pooled(B[ix])
    return float(pooled(A) - pooled(B)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def compare(da, db, thr):
    """da = the run, db = the control (`collect`) -> the record of `flip_<control>.json`"""
    rb, ra = rate(db, thr), rate(da, thr)
    tm = matched_thr(da, rb["true_pairs_t0_delivered"])
    rm = rate(da, tm)
    keys = ("difference", "lo95", "hi95")
    return dict(thr=thr, control=rb, run=ra, run_matched=rm, equal_threshold=dict(zip(keys, boot_diff(counts(da, thr), counts(db, thr)))), matched_delivery=dict(zip(keys, boot_diff(counts(da, tm), counts(db, thr)))),
                bootstrap=dict(unit="scene", scenes=len(da), resamples=N_BOOT, seed=BOOT_SEED, statistic="pooled flip rate, run - control"))


def selftest():
    one = {0: (np.array([0.9, 0.9, 0.05, 0.3]), np.array([0.9, 0.0, 0.5, 0.3]))}                         # at 0.1: delivered at t-1 [1, 1, 0, 1], at t [1, 0, 1, 1] -> 4 in either, 2 in exactly one
    r = rate(one, 0.1)
    assert (r["flips_one_only"], r["flips_total_either"], r["true_pairs_t0_delivered"], r["flip_rate"]) == (2, 4, 3, 0.5), r
    assert rate(one, 0.6)["flip_rate"] == 0.5 and rate(one, 0.6)["flips_total_either"] == 2                # at 0.6: [1, 1, 0, 0] and [1, 0, 0, 0]
    assert matched_thr(one, 2) == 0.9 and matched_thr(one, 3) == 0.3 and rate(one, matched_thr(one, 3))["true_pairs_t0_delivered"] == 3 and matched_thr(one, 99) == 0.05
    stable = {s: (np.full(10, 0.9), np.full(10, 0.9)) for s in range(6)}                                  # no flip at all
    flippy = {s: (np.full(10, 0.9), np.zeros(10)) for s in range(6)}                                      # every pair flips
    assert boot_diff(counts(stable, 0.1), counts(flippy, 0.1)) == (-1.0, -1.0, -1.0) and boot_diff(counts(stable, 0.1), counts(stable, 0.1)) == (0.0, 0.0, 0.0)
    rng = np.random.RandomState(1)                                                                        # a known difference of 0.2 with scatter over the scenes: the interval holds it, excludes 0, and the seed fixes it
    mk = lambda f: {s: (np.full(200, 0.9), np.where(rng.rand(200) < f, 0.0, 0.9)) for s in range(22)}  # noqa: E731
    a, b = mk(0.2), mk(0.4)
    d, lo, hi = boot_diff(counts(a, 0.1), counts(b, 0.1))
    assert lo < d < hi < 0 and abs(d + 0.2) < 0.03 and 0.01 < hi - lo < 0.08 and (d, lo, hi) == boot_diff(counts(a, 0.1), counts(b, 0.1)), (d, lo, hi)
    # matched delivery: the same pairs as the control but half of them weak (0.15) and flipping, plus weak junk; at the control's count the run keeps its 5 strong pairs and the difference vanishes
    ctrl = {0: (np.full(5, 0.9), np.full(5, 0.9))}
    arm = {0: (np.array([0.9] * 5 + [0.15] * 5), np.array([0.9] * 5 + [0.0] * 5))}
    c = compare(arm, ctrl, 0.1)
    assert c["run"]["flip_rate"] == 0.5 and c["run_matched"]["true_pairs_t0_delivered"] == 5 == c["control"]["true_pairs_t0_delivered"] and c["run_matched"]["thr"] == 0.9
    assert c["equal_threshold"]["difference"] == 0.5 and c["matched_delivery"]["difference"] == 0.0
    must = dict(a)
    must.pop(0)
    try:
        boot_diff(counts(must, 0.1), counts(b, 0.1))
        raise RuntimeError("two dumps with different scenes must be refused")
    except AssertionError:
        pass
    print("jflip selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c_ in ("rate", "compare"):
        s = sub.add_parser(c_)
        s.add_argument("--run", required=True)
        s.add_argument("--seed", type=int, required=True)
        s.add_argument("--thr", type=float, default=JTM.THR)
        if c_ == "compare":
            s.add_argument("--control", required=True)
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "rate":
        print(json.dumps(dict(run=a.run, **rate(collect(a.run, a.seed), a.thr)), indent=1))
    else:
        out = dict(run=a.run, control_run=a.control, seed=a.seed, **compare(collect(a.run, a.seed), collect(a.control, a.seed), a.thr))
        print(json.dumps(out, indent=1))
        print(f"true pairs delivered at t-1: control {out['control']['true_pairs_t0_delivered']} (p >= {a.thr}), run {out['run']['true_pairs_t0_delivered']} (p >= {a.thr}), run at matched delivery {out['run_matched']['true_pairs_t0_delivered']} (p >= {out['run_matched']['thr']:.4f})")
        json.dump(out, open(os.path.join(ROOT, "runs", a.run, f"flip_{a.control}.json"), "w"), indent=1)
