"""E318: is the learned position scale of a keypoint worth anything?  Scratch scorer of the "informative" clause of the E318 card (experiments/E318_end-position-uncertainty/experiment.md, section 0).  CPU, VAL 873.

    wsl bash tools/env/_py14.sh sigma_check.py e318_u_s0 [--thr 0.1] [--boot 1000]          # reads runs/<run>/ends_val.pkl (after `jends_eval.py dump --run e318_u_s0`): keys kp_pos, kp_dir, kp_p, valid, logit, allowed, logsig
    wsl bash tools/env/_py14.sh sigma_check.py e312_p_s0 --as-sigma range                    # self-test on a dump WITHOUT a scale: `range` (or `kp_p`) plays the role of the scale (a null, not a sigma)
    wsl bash tools/env/_py14.sh sigma_check.py selftest

The definitions (written into the card; nothing else is read):
  keypoint end error  every VALID keypoint of a kind (the first K of the 2K arrays are entries, the rest exits; a keypoint of the dump is a real peak of the heat) that has a GT end of its kind within 4 m and 30 degrees
                      (`qendnet.MATCH_POS / MATCH_ANG`, the matching of `jends.geom_loss`: many to one, the NEAREST such GT end) -> error = the distance to that end in metres. Errors beyond 4 m are not matched and not in the statistic.
  Spearman            rank correlation (average ranks for ties) of the unsure score of the keypoint with its end error, pooled over entries and exits; sigma = exp(logsig). A scene bootstrap gives the interval.
  delivered pair      (entry i, exit j) with allowed[i, j] and sigmoid(logit[i, j]) >= thr (the read-out of `qends_eval.cmd_score`, thr 0.1 in the cards); TRUE when the two keypoints are matched to the two ends of ONE GT pair by the STRICT one-to-one
                      matching of `jrule_control._match` (greedy by distance, 4 m / 30 degrees, every keypoint and every GT end used at most once; all valid keypoints take part: --kp-thr 0 as the read-out; `jpair_pr.py` uses 0.5).
                      The GT pairs and ends come from `lane_gt_data.QData("val_clean")` (`ent`, `ext`, `pairs`).
  fifths              the delivered pairs of all 873 frames pooled, sorted by the unsure score of the pair (sum of the two sigmas): the most CONFIDENT fifth = the 20 % smallest scores, the most UNSURE fifth = the 20 % largest;
                      precision = TRUE / delivered in each fifth; the difference (confident - unsure) is the number of the card (>= 0.03). The 95 % interval is a scene bootstrap (22 VAL scenes, frames of a scene stay together,
                      the fifths are re-cut in every resample). The card's rule reads the point estimate; the interval is printed next to it.
  nulls               the same two statistics with the scale replaced by free information: `range` (distance of the keypoint from the ego; pair: the sum of the two), `-kp_p` (heat probability at the peak turned into an unsure score;
                      pair: -(p_i + p_j)), and for the fifths the pair logit (`-logit`: the head's own confidence, the reference the scale has to be compared with, not a null of the scale).
"""
import argparse
import json
import math
import os
import pickle
import sys

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for sub in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", "tools/viz", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))

MATCH_POS, MATCH_ANG = 4.0, 30.0       # = qendnet.MATCH_POS / MATCH_ANG (asserted in main(): the card's matching is the loss' matching)
CLAMP = 2.95                            # JEnds clamps log sigma to [-3, 3]: a value this close is "at the clamp"
FAR = 35.0                              # m: "near" / "beyond 35 m" as in the maps cards


# ------------------------------------------------------------------ statistics (numpy only)
def avg_ranks(x):
    """ranks 1..n with the average rank for ties"""
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    u, inv, cnt = np.unique(x[o], return_inverse=True, return_counts=True)
    last = np.cumsum(cnt)
    avg = (last - cnt + 1 + last) / 2.0
    r = np.empty(len(x))
    r[o] = avg[inv]
    return r


def spearman(a, b):
    ra, rb = avg_ranks(a), avg_ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def fifths(u, true, frac=0.2):
    """(precision of the 20 % smallest scores, precision of the 20 % largest) of the pooled pairs; u: unsure score (higher = less sure), true: bool"""
    n = len(u)
    k = int(round(frac * n))
    if k < 1:
        return float("nan"), float("nan")
    o = np.argsort(u, kind="stable")
    return float(true[o[:k]].mean()), float(true[o[n - k:]].mean())


def scene_index(scene):
    uniq = sorted(set(scene.tolist()))
    return uniq, {s: np.where(scene == s)[0] for s in uniq}


def boot_idx(uniq, where, rng):
    return np.concatenate([where[uniq[i]] for i in rng.randint(0, len(uniq), len(uniq))])


def interval(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) else (float("nan"), float("nan"))


# ------------------------------------------------------------------ the per-keypoint and the per-pair tables of one dump
def match_end(kp, kd, valid, G):
    """(has, err, idx) per keypoint of one kind: the nearest GT end within MATCH_POS / MATCH_ANG (many to one, `jends.geom_loss`); G: (n, 4) x, y, tx, ty"""
    n = len(kp)
    if not len(G):
        return np.zeros(n, bool), np.zeros(n), np.zeros(n, int)
    dk = np.linalg.norm(kp[:, None, :] - G[None, :, :2], axis=-1)
    ang = (kd[:, None, :] * G[None, :, 2:]).sum(-1)
    ok = (dk <= MATCH_POS) & (ang >= math.cos(math.radians(MATCH_ANG))) & valid[:, None]
    a = np.where(ok, dk, 1e6).argmin(1)
    return ok.any(1), dk[np.arange(n), a], a


def tables(frames, gt, scene_of, thr, kp_thr, match_strict):
    """frames: the dump (list of dicts); gt: {token: dict(ent, ext, pairs)}; scene_of: {token: scene id}; match_strict(kp, kd, valid, G) -> (n_gt,) keypoint index or -1.
    -> (kp, pr): dicts of numpy arrays; kp: one row per matched keypoint (scene, kind, err, logsig, rng, p), pr: one row per delivered pair (scene, true, ls_i, ls_j, rng_i, rng_j, p_i, p_j, logit)"""
    kp = {k: [] for k in ("scene", "kind", "err", "logsig", "rng", "p")}
    pr = {k: [] for k in ("scene", "true", "ls_i", "ls_j", "rng_i", "rng_j", "p_i", "p_j", "logit")}
    for f in frames:
        t = f["token"]
        g = gt[t]
        K = f["logit"].shape[0]
        pos, dirs, p = np.asarray(f["kp_pos"], float), np.asarray(f["kp_dir"], float), np.asarray(f["kp_p"], float)
        valid = np.asarray(f["valid"], bool)
        ls = np.asarray(f["logsig"], float) if f.get("logsig") is not None else np.full(2 * K, np.nan)
        rng = np.linalg.norm(pos, axis=1)
        for k, G in ((0, g["ent"]), (1, g["ext"])):
            sl = slice(k * K, (k + 1) * K)
            has, err, _ = match_end(pos[sl], dirs[sl], valid[k], np.asarray(G, float).reshape(-1, 4))
            for i in np.where(has)[0]:
                kp["scene"].append(scene_of[t]), kp["kind"].append(k), kp["err"].append(err[i]), kp["logsig"].append(ls[k * K + i]), kp["rng"].append(rng[k * K + i]), kp["p"].append(p[k * K + i])
        gp = {(int(a), int(b)) for a, b in np.asarray(g["pairs"]).reshape(-1, 2)}
        ve, vx = valid[0] & (p[:K] >= kp_thr), valid[1] & (p[K:] >= kp_thr)
        ma, mb = match_strict(pos[:K], dirs[:K], ve, np.asarray(g["ent"], float).reshape(-1, 4)), match_strict(pos[K:], dirs[K:], vx, np.asarray(g["ext"], float).reshape(-1, 4))
        inv_a, inv_b = {int(i): a for a, i in enumerate(ma) if i >= 0}, {int(j): b for b, j in enumerate(mb) if j >= 0}
        prob = 1.0 / (1.0 + np.exp(-np.asarray(f["logit"], float))) * np.asarray(f["allowed"], bool)
        ii, jj = np.nonzero(prob >= thr)
        for i, j in zip(ii, jj):
            pr["scene"].append(scene_of[t]), pr["true"].append((inv_a.get(int(i), -1), inv_b.get(int(j), -1)) in gp)
            pr["ls_i"].append(ls[i]), pr["ls_j"].append(ls[K + j]), pr["rng_i"].append(rng[i]), pr["rng_j"].append(rng[K + j]), pr["p_i"].append(p[i]), pr["p_j"].append(p[K + j]), pr["logit"].append(float(f["logit"][i, j]))
    return ({k: np.asarray(v) for k, v in kp.items()}, {k: np.asarray(v) for k, v in pr.items()})


def strict_match(kp, kd, valid, G):
    import jrule_control as JR
    return JR._match(kp, kd, valid, G)


# ------------------------------------------------------------------ the report
def report(kp, pr, n_boot, seed=0, as_sigma=None, name=""):
    """-> dict of the numbers; prints them"""
    out = {"run": name}
    has_sig = np.isfinite(kp["logsig"]).any()
    # unsure scores per keypoint and per pair
    ks, ps = {}, {}
    if has_sig and as_sigma is None:
        ks["sigma"], ps["sigma"] = np.exp(kp["logsig"]), np.exp(pr["ls_i"]) + np.exp(pr["ls_j"])
    elif as_sigma == "range":
        ks["sigma"], ps["sigma"] = kp["rng"], pr["rng_i"] + pr["rng_j"]
    elif as_sigma == "kp_p":
        ks["sigma"], ps["sigma"] = -kp["p"], -(pr["p_i"] + pr["p_j"])
    if "sigma" in ks and as_sigma:
        print(f"STAND-IN: the dump has no scale (or --as-sigma was given): 'sigma' below is `{as_sigma}` as an unsure score. This tests the script, it says nothing about E318.")
    ks["range"], ps["range"] = kp["rng"], pr["rng_i"] + pr["rng_j"]
    ks["-kp_p"], ps["-kp_p"] = -kp["p"], -(pr["p_i"] + pr["p_j"])
    ps["-logit"] = -pr["logit"]
    n_kp, n_pr = len(kp["err"]), len(pr["true"])
    near = kp["rng"] < FAR
    print(f"keypoints matched to a GT end (4 m / 30 degrees, nearest end): {n_kp} (entries {int((kp['kind'] == 0).sum())}, exits {int((kp['kind'] == 1).sum())}); end error mean {kp['err'].mean():.2f} m, median {np.median(kp['err']):.2f} m")
    print(f"delivered pairs: {n_pr}; precision {pr['true'].mean():.4f}; scenes {len(set(kp['scene'].tolist()))}")
    out.update(n_keypoints=n_kp, n_pairs=n_pr, pair_precision=float(pr["true"].mean()), end_error_mean=float(kp["err"].mean()))
    rng = np.random.RandomState(seed)
    uk, wk = scene_index(kp["scene"])
    up, wp = scene_index(pr["scene"])
    draws = [(boot_idx(uk, wk, rng), boot_idx(up, wp, rng)) for _ in range(n_boot)]
    print("Spearman(unsure score, end error), pooled; 95 % scene-bootstrap interval  [rule: sigma >= 0.3]")
    out["spearman"] = {}
    for nm, u in ks.items():
        r = spearman(u, kp["err"])
        lo, hi = interval([spearman(u[ik], kp["err"][ik]) for ik, _ in draws])
        rk = {k_: spearman(u[kp["kind"] == k_], kp["err"][kp["kind"] == k_]) for k_ in (0, 1)}
        print(f"  {nm:8s} {r:+.3f}  [{lo:+.3f}, {hi:+.3f}]   entries {rk[0]:+.3f}, exits {rk[1]:+.3f}   near (<{FAR:.0f} m) {spearman(u[near], kp['err'][near]):+.3f}, beyond {spearman(u[~near], kp['err'][~near]):+.3f}")
        out["spearman"][nm] = dict(rho=r, lo=lo, hi=hi, entries=rk[0], exits=rk[1])
    print("precision of the delivered pairs: most confident fifth / most unsure fifth / difference; 95 % interval  [rule: sigma >= +0.03]")
    out["fifths"] = {}
    for nm, u in ps.items():
        c, s = fifths(u, pr["true"])
        d = [fifths(u[ip], pr["true"][ip]) for _, ip in draws]
        lo, hi = interval([a - b for a, b in d])
        print(f"  {nm:8s} {c:.4f} / {s:.4f} / {c - s:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
        out["fifths"][nm] = dict(confident=c, unsure=s, diff=c - s, lo=lo, hi=hi)
    if "sigma" in ks and not as_sigma:
        ls = kp["logsig"]
        allk = out["clamp"] = dict(at_clamp=float((np.abs(ls) >= CLAMP).mean()), median=float(np.median(ls)), std=float(ls.std()), within_0_05_of_median=float((np.abs(ls - np.median(ls)) <= 0.05).mean()))
        print(f"log sigma of the matched keypoints: median {allk['median']:+.2f}, std {allk['std']:.2f}, at the clamp (|s| >= {CLAMP}) {100 * allk['at_clamp']:.1f} %, within 0.05 of the median {100 * allk['within_0_05_of_median']:.1f} %   [kill: at the clamp > 90 %]")
        sg = np.exp(ls)
        print(f"mean sigma (m): near (< {FAR:.0f} m) {sg[near].mean():.2f}, beyond {sg[~near].mean():.2f}; mean end error: near {kp['err'][near].mean():.2f} m, beyond {kp['err'][~near].mean():.2f} m ({int(near.sum())} / {int((~near).sum())} keypoints)")
        out["sigma_by_range"] = dict(near=float(sg[near].mean()), beyond=float(sg[~near].mean()), err_near=float(kp["err"][near].mean()), err_beyond=float(kp["err"][~near].mean()))
        r, d, rd = out["spearman"]["sigma"]["rho"], out["fifths"]["sigma"]["diff"], out["fifths"]["range"]["diff"]
        ok = r >= 0.3 and d >= 0.03                                                                   # the range split `rd` is REPORTED, not a condition (second critic pass, 2026-11-14)
        out["informative"] = bool(ok)
        print(f"READING of the card's clauses: Spearman {r:+.3f} {'>=' if r >= 0.3 else '<'} 0.3; fifths difference {d:+.4f} {'>=' if d >= 0.03 else '<'} 0.03 and (range split reported: {'above' if d > rd else 'not above'}) the range split {rd:+.4f}  ->  "
              f"{'informative on this seed' if ok else 'NOT informative on this seed'} (the card needs both seeds; the range split is printed for reference, it is not a condition)")
    return out


# ------------------------------------------------------------------ data
def load_gt(cache):
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    import lane_gt_data as LD
    LD.QData.DEFAULT_JUNCTION = True
    ds = LD.QData("val_clean", map_mode="error", pixels=False)
    assert ds.missing == 0, f"{ds.missing} VAL frames have no stored lane GT"
    gt = {}
    for i in range(len(ds)):
        it = ds[i]
        gt[it["token"]] = dict(ent=np.asarray(it["ent"], np.float64).reshape(-1, 4), ext=np.asarray(it["ext"], np.float64).reshape(-1, 4), pairs=np.asarray(it["pairs"], np.int64).reshape(-1, 2))
    pickle.dump(gt, open(cache, "wb"))
    return gt


def main(a):
    import qendnet as QEN
    assert (QEN.MATCH_POS, QEN.MATCH_ANG) == (MATCH_POS, MATCH_ANG), "the matching constants of the loss changed: update this script and the card"
    import qgraph_view as QV
    src = os.path.join(ROOT, a.run) if a.run.endswith(".pkl") else os.path.join(ROOT, "runs", a.run, "ends_val.pkl")   # a path (from the problem folder, or absolute) to a dump works too: a test dump with a planted scale
    frames = pickle.load(open(src, "rb"))
    assert len(frames) == 873, f"{a.run}: the dump has {len(frames)} frames (a --max test dump?)"
    gt = load_gt(a.gt_cache)
    scene_of = {t: sid for sid, (name, toks) in QV.val_scenes().items() for t in toks}
    missing = [f["token"] for f in frames if f["token"] not in gt or f["token"] not in scene_of]
    assert not missing, f"{len(missing)} dump frames without GT or scene, first {missing[:2]}"
    kp, pr = tables(frames, gt, scene_of, a.thr, a.kp_thr, strict_match)
    assert len(pr["true"]), f"{a.run}: no delivered pair at p >= {a.thr}"
    out = report(kp, pr, a.boot, as_sigma=a.as_sigma, name=a.run)
    out.update(thr=a.thr, kp_thr=a.kp_thr, boot=a.boot, as_sigma=a.as_sigma)
    dst = os.path.join(ROOT, "temp", "e318", f"sigma_check_{os.path.basename(a.run)[:-4] if a.run.endswith('.pkl') else a.run}{'_as_' + a.as_sigma if a.as_sigma else ''}.json")
    json.dump(out, open(dst, "w"), indent=1)
    print("written", os.path.relpath(dst, ROOT).replace(os.sep, "/"))


# ------------------------------------------------------------------ self-test (synthetic, no data)
def selftest():
    # average ranks and Spearman with ties
    assert list(avg_ranks([10, 20, 20, 30])) == [1.0, 2.5, 2.5, 4.0]
    assert abs(spearman([1, 2, 3, 4], [2, 4, 6, 9]) - 1.0) < 1e-12 and abs(spearman([1, 2, 3, 4], [9, 6, 4, 2]) + 1.0) < 1e-12
    x, y = np.array([1, 1, 2, 2, 3, 3.0, 5, 4]), np.array([1, 2, 1, 2, 3, 4.0, 9, 2])
    assert math.isnan(spearman([1, 1, 1], [1, 2, 3]))
    try:
        from scipy.stats import spearmanr
        assert abs(spearman(x, y) - spearmanr(x, y)[0]) < 1e-12, (spearman(x, y), spearmanr(x, y)[0])                  # the tie handling against scipy
    except ImportError:
        pass
    # fifths: ten pairs, k = 2
    u = np.arange(10.0)
    true = np.array([1, 1, 0, 1, 1, 0, 0, 1, 0, 0], bool)
    assert fifths(u, true) == (1.0, 0.0) and fifths(-u, true) == (0.0, 1.0)
    # the matching of a keypoint to its nearest GT end: 4 m and 30 degrees, valid only, nearest wins, many to one
    G = np.array([[0.0, 0.0, 1.0, 0.0], [3.0, 0.0, 1.0, 0.0]])
    kp = np.array([[1.0, 0.0], [5.5, 0.0], [0.0, 3.9], [1.0, 0.0], [2.9, 0.0]])
    kd = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [math.cos(math.radians(40)), math.sin(math.radians(40))], [1.0, 0.0]])
    va = np.array([True, True, True, True, False])
    has, err, idx = match_end(kp, kd, va, G)
    assert list(has) == [True, True, True, False, False], has                    # (5.5, 0) is 2.5 m from the second end, (0, 3.9) is within 4 m of the first, 40 degrees is out, an invalid keypoint is out
    assert abs(err[0] - 1.0) < 1e-9 and idx[0] == 0 and idx[1] == 1 and abs(err[1] - 2.5) < 1e-9 and abs(err[2] - 3.9) < 1e-9
    # one frame through `tables`: GT entries 0, 1 and exit 0, true pair 0 -> 0; keypoints: entries 0, 1 (+ an invalid one), exits 0, 1 (two keypoints near the one GT exit) (+ an invalid one)
    gt = {"t": dict(ent=np.array([[0.0, 0.0, 1.0, 0.0], [10.0, 0.0, 1.0, 0.0]]), ext=np.array([[20.0, 0.0, 1.0, 0.0]]), pairs=np.array([[0, 0]]))}
    K = 3
    pos = np.array([[0.5, 0.0], [10.5, 0.0], [50.0, 50.0], [20.5, 0.0], [20.8, 0.0], [60.0, 0.0]])
    dr = np.tile([1.0, 0.0], (6, 1))
    logit = np.full((K, K), -9.0)
    logit[0, 0], logit[1, 0], logit[0, 1] = 3.0, 1.0, 2.0
    fr = dict(token="t", kp_pos=pos, kp_dir=dr, kp_p=np.array([0.9, 0.8, 0.0, 0.7, 0.6, 0.0]), valid=np.array([[1, 1, 0], [1, 1, 0]], bool), logit=logit, allowed=np.ones((K, K), bool),
              logsig=np.array([0.0, 1.0, 0.0, -1.0, 2.0, 0.0]))
    kp_t, pr_t = tables([fr], gt, {"t": 7}, 0.1, 0.0, strict_match)
    assert len(kp_t["err"]) == 4 and sorted(np.round(kp_t["err"], 1).tolist()) == [0.5, 0.5, 0.5, 0.8], kp_t["err"]     # the second exit keypoint also has the GT exit within 4 m (many to one)
    # delivered: (0,0) TRUE; (1,0) false (entry 1 is GT entry 1, the GT pair is 0 -> 0); (0,1) false (the strict matching gives the one GT exit to the nearer exit keypoint 0)
    assert sorted(zip(pr_t["logit"].tolist(), pr_t["true"].tolist())) == [(1.0, False), (2.0, False), (3.0, True)], (pr_t["logit"], pr_t["true"])
    rep = report(kp_t, pr_t, 20, as_sigma="range", name="synthetic")
    assert rep["n_pairs"] == 3 and abs(rep["pair_precision"] - 1 / 3) < 1e-12
    rep2 = report(kp_t, pr_t, 20, as_sigma=None, name="synthetic")                                           # the real-scale path: clamp share, sigma by range, the reading
    assert "clamp" in rep2 and "informative" in rep2 and rep2["clamp"]["at_clamp"] == 0.0, rep2
    # a planted relation: sigma follows the error and the truth of a pair follows sigma -> both clauses read clearly positive, a free quantity (range) does not
    r = np.random.RandomState(0)
    n = 3000
    err = r.gamma(2.0, 0.6, n)
    ls = np.log(err) + 0.3 * r.randn(n)
    assert spearman(np.exp(ls), err) > 0.7 and abs(spearman(r.uniform(5, 60, n), err)) < 0.1
    m = 4000
    u = np.exp(r.randn(m)) + np.exp(r.randn(m))
    c, s_ = fifths(u, r.rand(m) < 1 / (1 + 0.5 * u))
    assert c - s_ > 0.2, (c, s_)
    print("sigma_check selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="a run name (runs/<run>/ends_val.pkl) or `selftest`")
    ap.add_argument("--thr", type=float, default=0.1, help="the pair probability of the read-out (the cards: 0.1)")
    ap.add_argument("--kp-thr", type=float, default=0.0, help="keypoints with a heat probability below this do not take part in the strict matching (0 = all valid keypoints, as the read-out; jpair_pr.py: 0.5)")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--as-sigma", choices=("range", "kp_p"), default=None, help="self-test on a dump without a scale: this free quantity plays the sigma")
    ap.add_argument("--gt-cache", default=os.path.join(ROOT, "temp", "e318", "gt_val.pkl"), help="the GT ends and pairs of the 873 VAL frames, built once (5 - 8 minutes on the loaded machine) from QData")
    a = ap.parse_args()
    if a.run == "selftest":
        selftest()
    else:
        main(a)
