"""one-off (E274 / report): pixel-map results of the E273 head per layer on 300 VAL frames, pooled over the frames, a cell counts when the other mask has a cell within one cell (the tolerance of the pictures)

    wsl bash tools/env/_py14.sh pixel_eval.py e273_p_s0 e273_p_s1 [n]
Layers: lane centres (1 m), events split / merge / stop / exit / start / end (1 m heat maps, peak > 0.5 against GT peak > 0.5), lane dividers, road dividers, road borders (0.5 m), road and intersection (2 m).
"""
import json
import os
import sys

import numpy as np
import torch
from scipy.ndimage import maximum_filter

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "training/train", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgates as QG  # noqa: E402
import qpredict as QP  # noqa: E402

n = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 300
runs = [a for a in sys.argv[1:] if not a.isdigit()]
EV = ("split", "merge", "stop", "exit", "start", "end")
LN = ("lane_divider", "road_divider", "boundary")
res = {}
for run in runs:
    model = QP.load_model(os.path.join(ROOT, "runs", run), "cuda")
    cnt = {}

    def add(name, p, t, tol):
        near = (lambda m: maximum_filter(m, size=2 * tol + 1)) if tol else (lambda m: m)
        tp_p = (p & near(t)).sum()
        fp = (p & ~near(t)).sum()
        tp_r = (t & near(p)).sum()
        fn = (t & ~near(p)).sum()
        c = cnt.setdefault(name, np.zeros(4))
        c += (tp_p, fp, tp_r, fn)
    with torch.no_grad():
        for b in QG._batches("exact", n):
            P = model.pyr(b["x"].to("cuda").float())
            c1 = P["centre1"].float().sigmoid().cpu().numpy()
            l05 = P["lines05"].float().sigmoid().cpu().numpy()
            r2 = P["road2"].float().sigmoid().cpu().numpy()
            for i in range(c1.shape[0]):
                g05, m05 = b["g05"][i].numpy(), b["m05"][i].numpy() > 0
                gc = np.zeros((100, 100), bool)
                pts = np.clip(np.floor(g05[m05] + 50.0).astype(int), 0, 99)
                gc[pts[:, 0], pts[:, 1]] = True
                add("lane centres (1 m)", c1[i, 0] > 0.5, gc, 1)
                for k, nm in enumerate(EV):
                    add(f"event: {nm} (1 m)", c1[i, 6 + k] > 0.5, b["events1"][i, k].numpy() > 0.5, 2)
                for k, nm in enumerate(LN):
                    add(f"{nm.replace('_', ' ')}s (0.5 m)", l05[i, k] > 0.5, b["lines_mask"][i, k].numpy() > 0.5, 1)
                add("road (2 m)", r2[i, 0] > 0.5, b["road2"][i, 0].numpy() > 0.5, 0)
                add("intersection (2 m, grown by 2 m)", r2[i, 1] > 0.5, b["road2"][i, 1].numpy() > 0.5, 0)
    out = {}
    for k, (tp_p, fp, tp_r, fn) in cnt.items():
        P_, R_ = tp_p / max(tp_p + fp, 1), tp_r / max(tp_r + fn, 1)
        out[k] = dict(P=float(P_), R=float(R_), F1=float(2 * P_ * R_ / max(P_ + R_, 1e-9)), n_gt=int(tp_r + fn))
    res[run] = out
    print(run, {k: round(v["F1"], 3) for k, v in out.items()}, flush=True)
os.makedirs(os.path.join(ROOT, "temp", "e274"), exist_ok=True)
json.dump(res, open(os.path.join(ROOT, "temp", "e274", "pixel_eval.json"), "w"), indent=1)
