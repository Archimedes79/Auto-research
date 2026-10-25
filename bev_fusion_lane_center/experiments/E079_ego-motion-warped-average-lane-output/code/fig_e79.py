"""Loop 7 / E79 figures: (1) failure-phase lane F1 per corruption type, single frame vs temporal fusion; (2) precision / recall of the map-free hypotheses by range and threshold.
Usage (gcar): fig_e79.py --run e79_test --keys single,ema_pL:0.7 --range e79_range"""
import argparse
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
RF = "/mnt/c/ML-Experiment/bev_fusion_lane_center/reports/source"
PERTS = ["dark", "cam_front_off", "fog", "cam_front3_off", "blur", "extr3", "radar_off"]
PNAME = {"dark": "night", "cam_front_off": "front cam off", "fog": "fog", "cam_front3_off": "3 fwd cams off", "blur": "blur", "extr3": "extrinsics 3 deg", "radar_off": "radar off"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="e79_test")
    ap.add_argument("--keys", default="single,ema_pL:0.7")
    ap.add_argument("--range", default="e79_range_nt")
    a = ap.parse_args()
    j = json.load(open(f"{OUT}/{a.run}/result.json"))
    keys = a.keys.split(",")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    w = 0.8 / len(keys)
    for k, key in enumerate(keys):
        v = [j["perts"][p][key]["fail"]["lane_F1"] for p in PERTS]
        ax[0].bar(np.arange(len(PERTS)) + k * w, v, w, label=key)
    ax[0].set_xticks(np.arange(len(PERTS)) + 0.4 - w / 2)
    ax[0].set_xticklabels([PNAME[p] for p in PERTS], rotation=25, fontsize=8)
    ax[0].set_ylabel("lane F1 @1 m, failure phase")
    ax[0].set_title("temporal fusion of the lane output (TEST, SD map, scene-constant bias 1 m)")
    ax[0].legend(fontsize=8)
    r = json.load(open(f"{OUT}/{a.range}/result.json"))
    for feat, col in (("clean", "tab:green"), ("blur", "tab:orange"), ("dark", "tab:red")):
        if feat not in r:
            continue
        P = [r[feat][t]["-50_50"]["P"] for t in ("0.4", "0.5", "0.6", "0.7")]
        R = [r[feat][t]["-50_50"]["R"] for t in ("0.4", "0.5", "0.6", "0.7")]
        ax[1].plot(R, P, "o-", color=col, label={"dark": "night", "blur": "blur", "clean": "clean"}[feat])
        for t, x, y in zip(("0.4", "0.5", "0.6", "0.7"), R, P):
            ax[1].annotate(t, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax[1].set_xlabel("recall @1 m")
    ax[1].set_ylabel("precision @1 m")
    ax[1].set_title("map-free lane hypotheses: threshold tau trades recall for precision")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{RF}/figures/e79_fusion.png", dpi=130)
    print("saved")


if __name__ == "__main__":
    main()
