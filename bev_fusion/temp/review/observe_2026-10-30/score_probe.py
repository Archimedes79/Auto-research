"""Score calibration of the box head and box velocity vs the dense velocity head, 60 val keyframes (review).
For vehicles in the front field of view: detections (score >= 0.05) matched greedily to the observed GT boxes (centre within 2 m).
usage: score_probe.py <run>"""
import math
import sys

import numpy as np

sys.path.insert(0, ".")
from evaluation import bev_probe as ob  # noqa: E402
from evaluation.bev_probe import NUSC_GROUPS, decode, gt_boxes  # noqa: E402
from evaluation.metrics import in_front  # noqa: E402

run = sys.argv[1] if len(sys.argv) > 1 else "runs/t1_r4_soa"
cfg, model, loss, step, ds, device = ob.load_all(run, "cpu")
grid = model.grid
by = {}
for k, tok in enumerate(ds.samples):
    by.setdefault(ds.nusc.get("sample", tok)["scene_token"], []).append(k)
idx = []
for v in by.values():
    idx += [v[len(v) // 3], v[2 * len(v) // 3]]
idx = [idx[int(j)] for j in np.linspace(0, len(idx) - 1, 60).round()]
tp, fp, n_gt, vel = [], [], 0, []
for i in idx:
    try:
        batch, res, gts, tap, vis, n = ob.run_frame(model, loss, cfg, ds, i, device, True)
    except KeyError:
        continue
    g = gts[0.0][0]
    gt = [b for b in gt_boxes(g) if b["group"] == "vehicle" and in_front(b)]
    dets = [d for d in decode({a: b[:1].float().cpu() for a, b in res["out"][0.0]["obj"].items()}, grid, NUSC_GROUPS, k=100,
                              threshold=0.05)[0] if d["group"] == "vehicle" and in_front(d)]
    n_gt += len(gt)
    left = list(gt)
    for d in sorted(dets, key=lambda d: -d["score"]):
        j = min(range(len(left)), key=lambda q: math.hypot(left[q]["cx"] - d["cx"], left[q]["cy"] - d["cy"]), default=None)
        if j is not None and math.hypot(left[j]["cx"] - d["cx"], left[j]["cy"] - d["cy"]) < 2.0:
            b = left.pop(j)
            tp.append(d["score"])
            vel.append((math.hypot(b["vx"], b["vy"]), math.hypot(d["vx"], d["vy"]), math.hypot(b["vx"] - d["vx"], b["vy"] - d["vy"])))
        else:
            fp.append(d["score"])
tp, fp = np.array(tp), np.array(fp)
print(f"vehicles in front view: GT {n_gt}, detections >= 0.05: {len(tp) + len(fp)} (TP {len(tp)}, FP {len(fp)})")
print("TP score quantiles 10/50/90: %.2f %.2f %.2f | FP score quantiles 50/90/99: %.2f %.2f %.2f" %
      (*np.percentile(tp, [10, 50, 90]), *np.percentile(fp, [50, 90, 99])))
best = None
for thr in np.arange(0.05, 0.9, 0.025):
    t, f = (tp >= thr).sum(), (fp >= thr).sum()
    prec, rec = t / max(t + f, 1), t / max(n_gt, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    if best is None or f1 > best[0]:
        best = (f1, thr, prec, rec)
    if abs(thr * 40 - round(thr * 40)) < 1e-6 and round(thr * 20) % 2 == 0:
        print(f"  thr {thr:.2f}: precision {prec:.2f} recall {rec:.2f} F1 {f1:.2f}")
print("best F1 %.2f at threshold %.3f (precision %.2f, recall %.2f)" % best)
v = np.array(vel)
mov = v[:, 0] > 1.0
print(f"box velocity on matched vehicles: moving GT {mov.sum()}: mean GT speed {v[mov, 0].mean():.2f}, predicted {v[mov, 1].mean():.2f}, "
      f"error |dv| {v[mov, 2].mean():.2f} (zero predictor {v[mov, 0].mean():.2f}); static GT {(~mov).sum()}: error {v[~mov, 2].mean():.2f}")
