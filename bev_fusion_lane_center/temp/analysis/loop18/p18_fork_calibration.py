"""Loop 18 / WP4 (literature gap 3: nobody measures the calibration of per-fork probabilities): reliability of the junction-head heat at predicted fork vertices as a probability that the fork is true (a ground-truth fork within 6 m).
Expected calibration error (10 bins) raw and after an isotonic recalibration fitted on the first half of the forks and applied to the second half (frames are in scene order, so the halves hold different scenes). Usage (gcar): p18_fork_calibration.py"""
import numpy as np

z = np.load("/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e18/fork_prune_val/fork_features.npz")


def ece(p, y, bins=10):
    e, n = 0.0, len(p)
    b = np.minimum((p * bins).astype(int), bins - 1)
    for k in range(bins):
        m = b == k
        if m.any():
            e += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return e


def pav(x, y):
    o = np.argsort(x)
    xs, ys = x[o], y[o].astype(float)
    blocks = [[ys[i], 1, xs[i], xs[i]] for i in range(len(ys))]  # mean, weight, xmin, xmax
    out = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] >= out[-1][0]:
            b2, b1 = out.pop(), out.pop()
            w = b1[1] + b2[1]
            out.append([(b1[0] * b1[1] + b2[0] * b2[1]) / w, w, b1[2], b2[3]])
    return np.array([b[2] for b in out]), np.array([b[0] for b in out])


for kind, nm in ((0, "split"), (1, "merge")):
    A = z[f"k{kind}"]
    heat, y = np.clip(A[:, 0], 0, 1), A[:, 5]
    h = len(A) // 2
    print(f"{nm}: {len(A)} predicted forks, {y.mean():.2f} true")
    edges = np.linspace(0, 1, 11)
    print("   heat bin: n / fraction true  " + " | ".join(f"{edges[i]:.1f}-{edges[i + 1]:.1f}: {int(((heat >= edges[i]) & (heat < edges[i + 1] + (i == 9))).sum())} / {y[(heat >= edges[i]) & (heat < edges[i + 1] + (i == 9))].mean() if ((heat >= edges[i]) & (heat < edges[i + 1] + (i == 9))).any() else float('nan'):.2f}" for i in range(10)))
    print(f"   ECE raw (all) {ece(heat, y):.3f}; second half raw {ece(heat[h:], y[h:]):.3f}", end="")
    xs, ms = pav(heat[:h], y[:h])
    cal = np.interp(heat[h:], xs, ms)
    print(f"; second half after isotonic fit on the first {ece(cal, y[h:]):.3f}")
