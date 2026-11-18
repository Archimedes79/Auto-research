"""Figures and assets of the week-3 Spurmitte report (2026-11-02): run with tools/env/_py14.sh reports/build/reports/fig_w3_status.py, then build_w3_status_report.py.
Reads only result files (metric cards runs/eval/<name>/val.json) and the prototype video; writes reports/results/week3/figures/*.png, copies the lane architecture drawing and
the prototype video to reports/results/week3/{figures,video}/ and cuts one frame out of the video (none of it is in git).
The BEV-system, end-to-end and tooling figures of the first version are kept for a separate report in reports/results/week3_framework/ (the first version of these scripts is git commit c0734ec)."""
import json
import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weekly_lib as W  # noqa: E402

R = W.ROOT
OUT = f"{R}/reports/results/week3"
FIG = f"{OUT}/figures"
VID = f"{OUT}/video"
os.makedirs(FIG, exist_ok=True)
os.makedirs(VID, exist_ok=True)
NAVY, BLUE, ORANGE, GREY = "#14213D", "#1d4ea3", "#c85f00", "#9aa4b2"


def card(name, split="val"):
    return json.load(open(f"{R}/runs/eval/{name}/{split}.json"))["metrics"]


def lane_systems():
    names = [("control", "control (E94, 628 ms)"), ("champion", "champion"), ("recall", "recall"), ("fast_recall", "fast_recall"), ("recall_sp0_dp", "prototype\nrecall_sp0_dp")]
    keys = [("Q", "Q"), ("topo_F1", "topo F1"), ("lanes_found", "lanes found"), ("R@1_far", "R@1 far")]
    cols = [GREY, "#6f8fd0", "#3b66c4", BLUE, ORANGE]
    fig, ax = plt.subplots(figsize=(8.6, 3.6), dpi=170)
    w = 0.16
    for i, (n, lab) in enumerate(names):
        m = card(n)
        vals = [m[k] for k, _ in keys]
        xs = [j + (i - 2) * w for j in range(len(keys))]
        ax.bar(xs, vals, w * 0.92, color=cols[i], label=lab)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=5.6, rotation=90, color=NAVY)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([l for _, l in keys], fontsize=9)
    ax.set_ylim(0.6, 0.86)
    ax.set_ylabel("VAL, 873 frames", fontsize=8)
    ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.17), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG}/w3_lane_systems.png")
    plt.close(fig)


def frames():
    try:
        import cv2
    except Exception as e:  # noqa: BLE001
        print("no cv2, frames skipped:", e)
        return
    src = f"{R}/runs/prototype/recall_sp0_dp_video/recall_sp0_dp_val.mp4"
    if not os.path.exists(src):
        print("missing", src)
        return
    shutil.copy2(src, f"{VID}/Week3_lane_prototype_recall_sp0_dp.mp4")
    cap = cv2.VideoCapture(src)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * 0.4))
    ok, fr = cap.read()
    if ok:
        cv2.imwrite(f"{FIG}/w3_lane_frame.png", fr)
    print("video", n, "frames", "frame cut" if ok else "no frame")
    p = f"{R}/runs/prototype/recall_sp0_dp_video/recall_sp0_dp_val_poster.png"
    if os.path.exists(p):
        shutil.copy2(p, f"{FIG}/w3_lane_poster.png")


if __name__ == "__main__":
    lane_systems()
    frames()
    shutil.copy2(f"{R}/training/architecture.svg", f"{FIG}/w3_arch_lane.svg")
    print("figures in", FIG, sorted(os.listdir(FIG)))
