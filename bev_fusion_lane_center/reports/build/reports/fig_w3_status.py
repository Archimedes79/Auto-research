"""Figures and assets of the week-3 status report (2026-11-02): run with tools/env/_py14.sh reports/build/reports/fig_w3_status.py, then build_w3_status_report.py.
Reads only result files: metric cards runs/eval/<name>/val.json, the gradient reports (temp/g1_route_j1.json, temp/gradient_report_f1.json), records/BOARD.md of bev_fusion;
writes reports/results/week3/figures/*.png, copies the two example videos to reports/results/week3/video/ and cuts one frame out of each (neither is in git)."""
import json
import os
import re
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weekly_lib as W  # noqa: E402

R = W.ROOT
REPO = os.path.dirname(R)
OUT = f"{R}/reports/results/week3"
FIG = f"{OUT}/figures"
VID = f"{OUT}/video"
os.makedirs(FIG, exist_ok=True)
os.makedirs(VID, exist_ok=True)
NAVY, BLUE, ORANGE, GREEN, GREY = "#14213D", "#1d4ea3", "#c85f00", "#1a7f3c", "#9aa4b2"


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
        b = ax.bar(xs, vals, w * 0.92, color=cols[i], label=lab)
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


def gradient():
    rep = json.load(open(f"{REPO}/temp/g1_route_j1.json"))
    g = rep["groups"]
    labels = [x["group"].split(" (")[0] for x in g]
    rn = [x["route_norm"] for x in g]
    bn = [x["builder_norm"] for x in g]
    cs = [x.get("cosine") for x in g]
    fig, ax = plt.subplots(figsize=(8.6, 3.5), dpi=170)
    ys = range(len(g))
    ax.barh([y + 0.2 for y in ys], bn, 0.38, color=GREY, label="builder loss")
    ax.barh([y - 0.2 for y in ys], rn, 0.38, color=ORANGE, label="route loss (old joint path)")
    for y, r, b, c in zip(ys, rn, bn, cs):
        txt = "route loss: exactly 0" if r == 0 else f"cosine {c:+.2f}"
        ax.text(max(r, b) + 0.15, y, txt, va="center", fontsize=7, color=NAVY)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("gradient norm per module group (2 TRAIN frames, CPU; indicative)", fontsize=8)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, 10.5)
    fig.tight_layout()
    fig.savefig(f"{FIG}/w3_gradient_route.png")
    plt.close(fig)


def board():
    txt = open(f"{REPO}/bev_fusion/records/BOARD.md", encoding="utf-8").read().splitlines()
    head = next(l for l in txt if l.startswith("| arm "))
    cols = [c.strip() for c in head.strip("|").split("|")][2:-1]
    arms = {}
    for l in txt:
        if l.startswith("| t8_bevfeat") or l.startswith("| t10_dual") or l.startswith("| t7_rgb") or l.startswith("| t7_res4"):
            cells = [c.strip() for c in l.strip("|").split("|")]
            arms[cells[0]] = cells[2:-1]
    fig, ax = plt.subplots(figsize=(8.6, 3.5), dpi=170)
    show = [("t8_bevfeat", ORANGE), ("t10_dual", BLUE)]
    w = 0.38
    for i, (a, c) in enumerate(show):
        vals = []
        for cell in arms[a]:
            m = re.match(r"([+-]\d+\.\d+)", cell)
            vals.append(float(m.group(1)) if m else 0.0)
        ax.bar([j + (i - 0.5) * w for j in range(len(vals))], vals, w * 0.92, color=c, label=a.replace("t8_bevfeat", "t8_bevfeat (BEVFusion features)").replace("t10_dual", "t10_dual (both feature sets)"))
    ax.axhline(0, color=NAVY, lw=0.6)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace(": ", "\n") for c in cols], fontsize=6.2)
    ax.set_ylabel("change against the base t8_ctl", fontsize=8)
    ax.legend(fontsize=7.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{FIG}/w3_bev_board.png")
    plt.close(fig)


def frames():
    try:
        import cv2
    except Exception as e:  # noqa: BLE001
        print("no cv2, frames skipped:", e)
        return
    jobs = [(f"{R}/runs/prototype/recall_sp0_dp_video/recall_sp0_dp_val.mp4", "Week3_lane_prototype_recall_sp0_dp.mp4", "w3_lane_frame.png"),
            (f"{REPO}/bev_fusion/runs/t11_vel_hurdle_acc2/prototype/qualitative/videos/scene-0003_dashboard.mp4", "Week3_bev_dashboard_scene-0003.mp4", "w3_bev_frame.png")]
    for src, dst, png in jobs:
        if not os.path.exists(src):
            print("missing", src)
            continue
        shutil.copy2(src, f"{VID}/{dst}")
        cap = cv2.VideoCapture(src)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * 0.4))
        ok, fr = cap.read()
        if ok:
            cv2.imwrite(f"{FIG}/{png}", fr)
        print(dst, n, "frames", "frame cut" if ok else "no frame")
    p = f"{R}/runs/prototype/recall_sp0_dp_video/recall_sp0_dp_val_poster.png"
    if os.path.exists(p):
        shutil.copy2(p, f"{FIG}/w3_lane_poster.png")
    for s in ("lane", "bev"):
        pass


def svgs():
    for src, dst in ((f"{R}/training/architecture.svg", "w3_arch_lane.svg"), (f"{REPO}/bev_fusion/training/architecture.svg", "w3_arch_bev.svg")):
        shutil.copy2(src, f"{FIG}/{dst}")


if __name__ == "__main__":
    lane_systems()
    gradient()
    board()
    frames()
    svgs()
    print("figures in", FIG, sorted(os.listdir(FIG)))
