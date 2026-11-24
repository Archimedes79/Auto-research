"""E381 card v2 (D126), CPU, scratch: the noise levels SP (position jitter, m per axis) and SE (existence-logit noise) of the arm n, fixed BEFORE any run.
On the TRAIN every-10th frames of the B' cache of seed S (the TRAIN frames of temp/e379/gap_tokens.py) over the valid tokens:
  SP: the mean distance of an assigned token to its GT node after jitter, |d u + n| with n ~ N(0, SP^2 I2) (the cache stores |d| only; under isotropic noise the distance depends on |d| only),
      bisected to the seed's VAL mean distance (all 873 VAL frames, measured here the same way as gap_tokens.py);
  SE: the AP of sigmoid(e + n) with n ~ N(0, SE^2) for 'assigned to a GT node', bisected to the seed's VAL AP.
Fixed RNG seeds (one draw of standard normals reused for every bisection step: the statistic is monotone in sigma). No head score is read.
usage: python calibrate_noise.py <seed>"""
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score

R = os.environ["BFLC_ROOT"]
for sub in ("training/data", "training/grid", "evaluation", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(R, sub)))
import lane_ptok_data as LP  # noqa: E402
import lane_tokcache as LC  # noqa: E402


def stats(split, seed, step):
    c = LC.TokCache(LP.cache_dir(split, seed, "e292_a"))
    e_all, y_all, gd_all = [], [], []
    for i in np.arange(0, len(c.tokens), step):
        v = np.array(c.f["valid"][i])
        e = np.array(c.f["e"][i], float)[v]
        g = np.array(c.f["gt_id"][i])[v] >= 0
        d = np.array(c.f["gd"][i], float)[v][g]
        e_all.append(e); y_all.append(g); gd_all.append(d[np.isfinite(d)])
    return np.concatenate(e_all), np.concatenate(y_all), np.concatenate(gd_all), len(np.arange(0, len(c.tokens), step))


def bisect(f, target, lo, hi, increasing, it=40):
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if (f(mid) < target) == increasing:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


seed = int(sys.argv[1])
e_t, y_t, d_t, n_t = stats("train", seed, 10)
e_v, y_v, d_v, n_v = stats("val", seed, 1)
ap = lambda e, y: average_precision_score(y, 1 / (1 + np.exp(-e)))  # noqa: E731
ap_t, ap_v, md_t, md_v = ap(e_t, y_t), ap(e_v, y_v), d_t.mean(), d_v.mean()
rng = np.random.RandomState(381000 + seed)
zp = rng.standard_normal((len(d_t), 2))
ze = rng.standard_normal(len(e_t))
mean_dist = lambda sp: float(np.mean(np.hypot(d_t + sp * zp[:, 0], sp * zp[:, 1])))  # noqa: E731
ap_noisy = lambda se: float(ap(e_t + se * ze, y_t))  # noqa: E731
SP = round(bisect(mean_dist, md_v, 0.0, 1.0, True), 3)
SE = round(bisect(ap_noisy, ap_v, 0.0, 5.0, False, it=25), 3)
print(f"seed {seed}: TRAIN every-10th {n_t} frames ({len(y_t)} valid tokens, {len(d_t)} assigned with a distance), VAL {n_v} frames ({len(y_v)} valid tokens)")
print(f"  clean: mean distance TRAIN {md_t:.4f} m / VAL {md_v:.4f} m; AP TRAIN {ap_t:.4f} / VAL {ap_v:.4f}")
print(f"  SP {SP} m per axis: mean distance after jitter {mean_dist(SP):.4f} m (target {md_v:.4f}); expected logged mean displacement SP sqrt(pi/2) = {SP * np.sqrt(np.pi / 2):.4f} m")
print(f"  SE {SE}: AP after noise {ap_noisy(SE):.4f} (target {ap_v:.4f}); expected logged mean |noise e| SE sqrt(2/pi) = {SE * np.sqrt(2 / np.pi):.4f}")
print(f"RESULT seed {seed} SP {SP} SE {SE}")
# v2 (critic of card v2, should-fix 4 + nits): the cache's gd exists only for tokens assigned within 1.5 m (lane_tokcache: r 1.5 m, heading cos >= 0.5; the heading is not perturbed), so the VAL target is a mean over
# distances <= 1.5 m: the same cut applied after the jitter (a jittered token beyond 1.5 m would be unassigned), SP bisected on that truncated mean; the mean p of assigned / unassigned tokens after the e noise
CUT = 1.5
dist = lambda sp: np.hypot(d_t + sp * zp[:, 0], sp * zp[:, 1])  # noqa: E731
mean_cut = lambda sp: float(dist(sp)[dist(sp) <= CUT].mean())  # noqa: E731
SPc = round(bisect(mean_cut, md_v, 0.0, 1.0, True), 3)
pe = 1 / (1 + np.exp(-(e_t + SE * ze)))
print(f"  v2 cut at {CUT} m: SP_cut {SPc} m per axis: truncated mean after jitter {mean_cut(SPc):.4f} m (target {md_v:.4f}); share of assigned TRAIN tokens jittered beyond {CUT} m: at SP {SP} {np.mean(dist(SP) > CUT):.4f}, "
      f"at SP_cut {np.mean(dist(SPc) > CUT):.4f}; clean VAL max distance {d_v.max():.3f} m")
print(f"  v2 mean p after the e noise (SE {SE}): assigned {pe[y_t].mean():.3f}, unassigned {pe[~y_t].mean():.3f} (clean TRAIN {(1 / (1 + np.exp(-e_t)))[y_t].mean():.3f} / {(1 / (1 + np.exp(-e_t)))[~y_t].mean():.3f}, "
      f"VAL {(1 / (1 + np.exp(-e_v)))[y_v].mean():.3f} / {(1 / (1 + np.exp(-e_v)))[~y_v].mean():.3f})")
print(f"  v2 chord direction between two tokens 2 m apart: the perpendicular difference of two jitters has SD sqrt(2) SP_cut = {np.sqrt(2) * SPc:.3f} m, i.e. about {np.degrees(np.arctan(np.sqrt(2) * SPc / 2)):.1f} degrees")
print(f"RESULT2 seed {seed} SP_cut {SPc} SE {SE}")
