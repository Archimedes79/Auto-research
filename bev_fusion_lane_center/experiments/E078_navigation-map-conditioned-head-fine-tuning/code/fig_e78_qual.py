"""Loop 7: qualitative examples of the SD-map head (E78): ground-truth lane centre lines, SD map (roads + route as given, with pose error sigma 1 m), predicted lane points and route points, clean and perturbed features.
Usage (gcar): fig_e78_qual.py --model e78_sd_a --out figures/e78_qualitative.png"""
import argparse
import os
import pickle
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dense_head as DH  # noqa: E402
import route_head3 as RH3  # noqa: E402
import sd_lib as SL  # noqa: E402
import temporal_eval as TE  # noqa: E402

RF = "/mnt/c/ML-Experiment/bev_fusion_lane_center/reports/source"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="e78_sd_a")
    ap.add_argument("--out", default="e78_qualitative.png")
    ap.add_argument("--feats", default="clean,dark,cam_front3_off")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    D = RH3.D
    P = dict(np.load(f"{D}/poses_val.npz"))
    SDm = RH3.sd_maps("val")
    RTv = pickle.load(open(f"{D}/route_gt_val.pkl", "rb"))
    GT = pickle.load(open(f"{D}/graph_gt_val.pkl", "rb"))
    gts = RH3.gt_points("val")
    rp = RH3.route_pts("val")
    net, mean, std = TE.load_model(a.model, "sd")
    rows_all = [i for i in RH3.SP["test"] if RTv[i] is not None and len(GT[i]["edges"]) >= 4]
    rng = np.random.RandomState(3)
    pick = sorted(rng.choice(rows_all, a.n, replace=False).tolist())
    feats = a.feats.split(",")
    fig, ax = plt.subplots(len(feats), a.n, figsize=(4.6 * a.n, 4.4 * len(feats)))
    ax = np.atleast_2d(ax)
    for fi, feat in enumerate(feats):
        if feat != "clean":
            have = set(np.load(f"{D}/pert/{feat}/test_rows.npy").tolist())
            rows = [r for r in pick if r in have] or [r for r in rows_all if r in have][: a.n]
        else:
            rows = pick
        out = TE.infer_rows(net, mean, std, feat, rows, P, SDm, 1.0, "sd")
        for k in range(a.n):
            axx = ax[fi, k]
            if k >= len(rows):
                axx.axis("off")
                continue
            r = rows[k]
            L0, L, R, _ = out[r]
            g = gts[int(r)]
            axx.plot(g[:, 1], g[:, 0], ".", color="0.8", ms=1.5, label="GT lanes")
            rg = rp[int(r)]
            if rg is not None:
                axx.plot(rg[:, 1], rg[:, 0], ".", color="tab:orange", ms=3, label="GT route")
            m = TE.sd_map_for(r, P, SDm, 1.0)
            ys, xs = np.nonzero(m[1])
            axx.plot((xs + 0.5) * 0.5 - 50, (ys + 0.5) * 0.5 - 50, ",", color="tab:blue", alpha=0.7)
            pl = DH.skeleton_points(L[None].float().cuda(), 0.5)[0]
            pr = DH.skeleton_points(R[None].float().cuda(), 0.5)[0]
            pl, pr = np.asarray(pl), np.asarray(pr)
            if len(pl):
                axx.plot(pl[:, 1], pl[:, 0], ".", color="tab:green", ms=2.5, label="predicted lanes (with SD map)")
            if len(pr):
                axx.plot(pr[:, 1], pr[:, 0], ".", color="tab:red", ms=2.5, label="predicted route")
            axx.set_xlim(-50, 50)
            axx.set_ylim(-50, 50)
            axx.set_aspect("equal")
            axx.set_title(f"{feat}, row {r}", fontsize=9)
            axx.tick_params(labelsize=7)
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, markerscale=4)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(f"{RF}/figures/{a.out}", dpi=100)
    print("saved", a.out, pick)


if __name__ == "__main__":
    main()
