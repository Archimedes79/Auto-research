"""blur of a recurrent state on the 50 x 50 2 m grid under repeated ego warps (bilinear, zeros outside) with the real VAL poses: a one-cell delta, a smooth ramp; numpy only"""
import json
import sys
import numpy as np

d = sys.argv[1]
prev = np.load(d + "/prev.npy")
RR = np.load(d + "/rel_R.npy").astype(float)
tt = np.load(d + "/rel_t.npy").astype(float)
N, HALF = 50, 50.0
c = -HALF + (np.arange(N) + 0.5) * 2.0
P = np.stack(np.meshgrid(c, c, indexing="ij"), -1)  # (N, N, 2) x(row), y(col)


def warp(s, R, t):
    q = (P - t) @ R  # R^T (p - t) as row vectors
    fr = (q[..., 0] + HALF) / 2.0 - 0.5  # fractional row index
    fc = (q[..., 1] + HALF) / 2.0 - 0.5
    r0, c0 = np.floor(fr).astype(int), np.floor(fc).astype(int)
    out = np.zeros_like(s)
    for dr in (0, 1):
        for dc in (0, 1):
            rr, cc = r0 + dr, c0 + dc
            w = (1 - np.abs(fr - rr)) * (1 - np.abs(fc - cc))
            ok = (rr >= 0) & (rr < N) & (cc >= 0) & (cc < N)
            out += np.where(ok, w * s[np.clip(rr, 0, N - 1), np.clip(cc, 0, N - 1)], 0.0)
    return out


rng = np.random.default_rng(0)
starts = [i for i in range(len(prev) - 12) if (prev[i + 1: i + 11] >= 0).all()]
res = {k: [] for k in (1, 2, 3, 5, 10)}
res_fw = {k: [] for k in (1, 2, 3, 5, 10)}
for i in rng.choice(starts, 60, replace=False):
    for cell in ((25, 25), (30, 20), (20, 30)):
        s = np.zeros((N, N)); s[cell] = 1.0
        for k in range(1, 11):
            j = i + k
            s = warp(s, RR[j], tt[j])
            if k in res:
                res[k].append(s.max() / 1.0)
                res_fw[k].append(s.sum())  # mass kept (zeros outside)
print(json.dumps({"peak_retained_median": {k: round(float(np.median(v)), 3) for k, v in res.items()}, "mass_retained_median": {k: round(float(np.median(v)), 3) for k, v in res_fw.items()}}))
sh = np.linalg.norm(tt[prev >= 0], axis=1)
print("ego shift per keyframe pair (m): median", round(float(np.median(sh)), 2), "p10", round(float(np.percentile(sh, 10)), 2), "p90", round(float(np.percentile(sh, 90)), 2), "; in cells (2 m):", round(float(np.median(sh)) / 2, 2))
