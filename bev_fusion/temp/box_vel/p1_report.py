"""t11 vel cause, P1: the per-frame and per-box statistics of a probe_features.py npz (active hidden units per branch at the supervised
centres and over all cells, the velocity / motion terms against their zero predictors, output sizes, static share of the targets).
Run (from bev_fusion/): ~/miniconda3/envs/gcar/bin/python temp/box_vel/p1_report.py temp/box_vel/out/t8_ctl_s0_val.npz [...]
"""
import sys

import numpy as np

BRANCHES = ("hm", "off", "size", "rot", "vel")


def report(path):
    z = np.load(path, allow_pickle=True)
    nf, nb = len(z["frame_n_box"]), len(z["group"])
    print(f"== {path}: step {int(z['step'])}, {nf} frames, {nb} supervised centres ({len(z['scenes'])} scenes)")
    for b in BRANCHES:
        ctr, allc, alive = z[f"frame_active_ctr_{b}"], z[f"frame_active_all_{b}"], z[f"frame_units_alive_{b}"]
        w = z["frame_n_box"]
        pooled = float(np.nansum(ctr * w) / w[np.isfinite(ctr)].sum())
        print(f"  {b:5s} active at the centres {pooled * 100:7.3f} % (frames with any: {int((ctr > 0).sum())}/{int(np.isfinite(ctr).sum())}), "
              f"over all cells {allc.mean() * 100:7.3f} %, units on at >= 1 cell {alive.mean():5.1f} / 64 (max {alive.max()})")
    zv = z["z_vel"].astype(np.float32)
    print(f"  vel pre-activation at the centres: max {zv.max():.3f}, mean {zv.mean():.3f}; (unit, centre) pairs > 0: {int((zv > 0).sum())} of {zv.size}")
    vt, v0 = z["frame_vel_term"], z["frame_vel_term_zero"]
    ok = np.isfinite(vt) & (z["frame_n_box"] > 0)
    print(f"  objhead.vel@+0.0 vs zero predictor: mean {vt[ok].mean():.4f} vs {v0[ok].mean():.4f}, max |diff| {np.abs(vt[ok] - v0[ok]).max():.2e}")
    mt, m0 = z["frame_motion"], z["frame_motion_zero"]
    ok = np.isfinite(mt)
    print(f"  motion@+0.0 vs v_hat = 0: mean {mt[ok].mean():.4f} vs {m0[ok].mean():.4f}, max |diff| {np.abs(mt[ok] - m0[ok]).max():.2e}")
    vel, vh, tv = z["vel"], z["vhat"], z["t_vel"]
    print(f"  output at the centres: max |vel| {np.linalg.norm(vel, axis=1).max():.2e} m/s, max |v_hat| {np.linalg.norm(vh, axis=1).max():.2e} m/s")
    groups = list(z["groups"])
    for gi, g in enumerate(groups):
        k = z["group"] == gi
        if k.any():
            sp = np.linalg.norm(tv[k], axis=1)
            print(f"  {g:12s} {int(k.sum()):6d} centres, static (< 0.5 m/s) {np.mean(sp < 0.5) * 100:5.1f} %, < 1 m/s {np.mean(sp < 1.0) * 100:5.1f} %, "
                  f"mean speed of the moving {sp[sp >= 0.5].mean() if (sp >= 0.5).any() else float('nan'):.2f} m/s")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
