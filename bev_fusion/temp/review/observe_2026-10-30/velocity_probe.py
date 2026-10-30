"""Velocity head against the zero predictor on the vehicle cells (moving and static), val and train frames, 30 scenes each
(review of the observability findings). usage: velocity_probe.py <run> [train|val]"""
import sys

import numpy as np
import torch

sys.path.insert(0, "evaluation")
import observe_run as ob  # noqa: E402
from train import make_datasets  # noqa: E402

run = sys.argv[1] if len(sys.argv) > 1 else "runs/t1_r4_soa"
split = sys.argv[2] if len(sys.argv) > 2 else "val"
cfg, model, loss, step, ds, device = ob.load_all(run, "cpu")
if split == "train":
    ds = make_datasets(cfg)[0]
grid, nusc = model.grid, ds.nusc
by = {}
for k, tok in enumerate(ds.samples):
    by.setdefault(nusc.get("sample", tok)["scene_token"], []).append(k)
mids = [v[len(v) // 2] for v in by.values()]
idx = [mids[int(j)] for j in np.linspace(0, len(mids) - 1, 30).round()]
E = {"mov_model": [], "mov_zero": [], "stat_model": [], "stat_zero": []}
frac_dead = []
for i in idx:
    try:
        batch, res, gts, tap, vis, n = ob.run_frame(model, loss, cfg, ds, i, device, False)
    except KeyError:
        continue
    g = gts[0.0][0]
    v = res["out"][0.0]["v_hat"][0].float().numpy()
    gv = np.asarray(g["velocity"], np.float32)
    veh = np.asarray(g["vehicle"], np.float32) > 0.5
    gsp = np.hypot(gv[0], gv[1])
    mov, stat = veh & (gsp > 1.0), veh & (gsp <= 1.0)
    if mov.any():
        E["mov_model"].append(np.abs(v[:, mov] - gv[:, mov]).mean())
        E["mov_zero"].append(np.abs(gv[:, mov]).mean())
        frac_dead.append(float((np.hypot(v[0][mov], v[1][mov]) < 0.5).mean()))
    if stat.any():
        E["stat_model"].append(np.abs(v[:, stat]).mean())
        E["stat_zero"].append(0.0)
print(f"{split}: frames with moving vehicles {len(E['mov_model'])}, with static vehicles {len(E['stat_model'])}")
print("moving vehicle cells: mean |v - GT| model %.2f, zero predictor %.2f m/s (per component)" %
      (np.mean(E["mov_model"]), np.mean(E["mov_zero"])))
print("static vehicle cells: mean |v| model %.2f m/s (zero predictor 0)" % np.mean(E["stat_model"]))
print("share of moving vehicle cells predicted below 0.5 m/s: %.0f %%" % (100 * np.mean(frac_dead)))
