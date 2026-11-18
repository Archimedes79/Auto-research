"""Figures of the loop-6 experiments (E73 study, E74 registration front end, E75 counterfactual routes, E76 trust).
Usage (gcar): fig_e73.py [--final NAME]  -> report_final/figures/e73_dev_scores.png, e73_pert_types.png, e74_registration.png, e75_alt.png, e76_trust.png (those with data)"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
RF = "/mnt/c/ML-Experiment/bev_fusion_lane_center/reports/source"
FIG = f"{RF}/figures"
PERTS = ["dark", "cam_front_off", "fog", "cam_front3_off", "blur", "extr3", "radar_off"]
PNAME = {"dark": "night", "cam_front_off": "front cam off", "fog": "fog", "cam_front3_off": "3 fwd cams off", "blur": "blur", "extr3": "extrinsics 3 deg", "radar_off": "radar off"}
STAGE_COL = {"anchor": "#9aa0a6", "1 data": "#1a73e8", "2 optimiser": "#e8710a", "3 layers": "#188038", "4 regularisation": "#a142f4", "4 loss": "#d93025", "5 trust": "#12b5cb", "6 weights": "#795548", "7 final": "#000000", "?": "#cccccc"}


def load(name):
    for fn in ("result.json", "result_eval.json"):
        p = f"{OUT}/e73_{name}/{fn}"
        if os.path.exists(p):
            return json.load(open(p))
    return None


def fig_scores():
    rows = json.load(open(f"{RF}/tables/e73_dev.json"))
    rows = [r for r in rows if r["stage"] != "?"]
    if not rows:
        return
    seeds = [r["score"] for r in rows if r["name"] in ("r0_s0", "r0_s1", "r0_s2")]
    fig, ax = plt.subplots(figsize=(11.5, 0.27 * len(rows) + 1.3))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r["score"] for r in rows], color=[STAGE_COL.get(r["stage"], "#ccc") for r in rows], height=0.7)
    for yy, r in zip(y, rows):
        ax.text(r["score"] + 0.0006, yy, f"{r['score']:.4f}", va="center", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['name']}  -  {r['desc'][:52]}" for r in rows], fontsize=7.2)
    lo = min(r["score"] for r in rows)
    ax.set_xlim(max(lo - 0.02, 0.5), max(r["score"] for r in rows) + 0.012)
    if len(seeds) >= 2:
        m, s = float(np.mean(seeds)), float(np.std(seeds, ddof=1))
        ax.axvspan(m - 2 * s, m + 2 * s, color="#fbbc04", alpha=0.25, label=f"R0 seeds: mean {m:.4f} +- 2 SD ({2 * s:.4f})")
        ax.axvline(m, color="#b06000", lw=1)
        ax.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("DEV score (mean of 8 numbers, see text)", fontsize=8)
    ax.set_title("E73: DEV score of every fine-tuning run (VAL scenes only)", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/e73_dev_scores.png", dpi=130)
    plt.close(fig)
    print("saved e73_dev_scores.png")


def fig_pert(names, labels):
    data = []
    for n in names:
        r = load(n)
        if not r:
            return
        ev = r.get("eval", {}).get("dev")
        if not ev:
            return
        data.append(ev)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
    w = 0.8 / len(names)
    for k, (key, ttl) in enumerate((("lane_F1", "lane F1 @1 m (exact map)"), ("route_success", "route success (exact map)"))):
        ax = axs[k]
        for i, ev in enumerate(data):
            vals = [ev[p]["exact"][key] for p in PERTS] + [ev["clean"]["exact"][key]]
            ax.bar(np.arange(len(vals)) + i * w, vals, w, label=labels[i])
        ax.set_xticks(np.arange(len(PERTS) + 1) + 0.4 - w / 2)
        ax.set_xticklabels([PNAME[p] for p in PERTS] + ["clean"], rotation=30, ha="right", fontsize=8)
        ax.set_title(ttl, fontsize=9)
        ax.set_ylim(0.3 if key == "route_success" else 0.6, 1.0)
        ax.grid(axis="y", alpha=0.3)
    axs[0].legend(fontsize=7.5)
    fig.suptitle("E73: DEV (VAL scenes) per perturbation type", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/e73_pert_types.png", dpi=130)
    plt.close(fig)
    print("saved e73_pert_types.png")


def fig_reg(models):
    """models: list of (label, run name of the E74 TEST evaluation)."""
    sig = [0, 1, 2, 3]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.6))
    geo = [0.926, 0.923, 0.883, 0.782]
    geo_noreg = [0.926, 0.640, 0.305, 0.153]
    axs[1].set_visible(True)
    for lab, name in models:
        r = load(name)
        if not r:
            continue
        c = r["eval"]["test"]["clean"]
        unreg = [c["exact"], c["s1"], c["s2"], c["s3"]]
        reg = [c["exact"], c["s1r"], c["s2r"], c["s3r"]]
        for ax, key in ((axs[0], "route_success"), (axs[1], "lane_F1")):
            ax.plot(sig, [u[key] for u in unreg], "o--", label=f"{lab}: map as localised")
            ax.plot(sig, [u[key] for u in reg], "o-", label=f"{lab}: map registered first")
    axs[0].plot(sig, geo, "k^-", label="geometric transfer (E59): registered")
    axs[0].plot(sig, geo_noreg, "k^:", label="geometric transfer (E59): not registered")
    axs[0].set_title("route success (TEST)", fontsize=9)
    axs[1].set_title("lane F1 @1 m (TEST)", fontsize=9)
    for ax in axs:
        ax.set_xlabel("map pose error sigma (m per axis; yaw 2 x sigma deg)")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("E74: registration in front of the end-to-end head", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/e74_registration.png", dpi=130)
    plt.close(fig)
    print("saved e74_registration.png")


def fig_alt(models):
    """models: list of (label, run name of the alt evaluation)."""
    labs, drv, alt = ["geometric transfer (E59)"], [0.887], [0.805]
    for lab, name in models:
        r = load(name)
        if not r:
            continue
        a = r["eval"]["alt"]["clean"]
        labs.append(lab)
        drv.append(a["given_drv"]["route_success@drv"])
        alt.append(a["given_alt"]["route_success@alt"])
    fig, ax = plt.subplots(figsize=(7.5, 3.3))
    x = np.arange(len(labs))
    ax.bar(x - 0.18, drv, 0.36, label="driven route given", color="#1a73e8")
    ax.bar(x + 0.18, alt, 0.36, label="alternative branch given", color="#e8710a")
    for xi, d, a_ in zip(x, drv, alt):
        ax.text(xi - 0.18, d + 0.01, f"{d:.2f}", ha="center", fontsize=8)
        ax.text(xi + 0.18, a_ + 0.01, f"{a_:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("route success against the GIVEN route")
    ax.legend(fontsize=8)
    ax.set_title("E75: does the head follow a changed goal? (256 TEST samples with a split, exact map)", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/e75_alt.png", dpi=130)
    plt.close(fig)
    print("saved e75_alt.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pert_runs", default="")
    ap.add_argument("--pert_labels", default="")
    ap.add_argument("--reg", default="")
    ap.add_argument("--alt", default="")
    a = ap.parse_args()
    fig_scores()
    if a.pert_runs:
        fig_pert(a.pert_runs.split(","), a.pert_labels.split(","))
    if a.reg:
        fig_reg([tuple(x.split("=")) for x in a.reg.split(",")])
    if a.alt:
        fig_alt([tuple(x.split("=")) for x in a.alt.split(",")])


if __name__ == "__main__":
    main()
