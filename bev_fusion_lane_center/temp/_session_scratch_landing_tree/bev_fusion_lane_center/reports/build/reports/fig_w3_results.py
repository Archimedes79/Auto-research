"""Week 3 result figures from the stored result files (no GPU): seed floor (E240 / E241), matched vertex budget (E255 / E244 / E236), registration replicate (E238),
metric by city (E252), timing (E243 / E253). Also writes reports/source/weekly/week3_numbers.json (every number the builder prints in tables, read from the files).
Writes reports/source/figures/w3_*.png.  Usage (gcar): fig_w3_results.py
A missing file (a run that did not finish) is skipped, never replaced by a typed number."""
import bflc_paths as BP  # project paths from the location of this checkout
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = BP.ROOT
FIG = f"{R}/reports/source/figures"
NUM = {}
C = dict(rc="#1f5fa8", eq="#8a8f98", ex="#2a9d4a", phd="#c85f00", bos="#1f5fa8", sg="#c97a12")


def card(name, tag=""):
    p = f"{R}/runs/eval/{name}/val{'_' + tag if tag else ''}.json"
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8-sig"))["metrics"]


def style(ax, title, ylabel=None):
    ax.set_title(title, fontsize=10, loc="left")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


# ---------------------------------------------------------------- 1 seed floor (E240) and early checkpoints (E241)
def fig_seed_floor():
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.0))
    seeds = [s for s in range(4) if card(f"e24_j1rc_s{s}") and card(f"e24_j1eq_s{s}")]
    NUM["e240_seeds"] = seeds
    for ax, key, title in ((axs[0], "Q", "Q (VAL 873, 14.3 - 15.2 vertices / 100 m)"), (axs[1], "lanes_found", "lanes found")):
        for k, (arm, lab) in enumerate((("eq", "j1eq (recall weight 1)"), ("rc", "j1rc (recall weight 3)"))):
            v = [card(f"e24_j1{arm}_s{s}")[key] for s in seeds]
            ax.scatter(np.full(len(v), k) + np.linspace(-0.12, 0.12, len(v)), v, s=45, color=C[arm], zorder=3)
            ax.plot([k - 0.25, k + 0.25], [np.mean(v)] * 2, color=C[arm], lw=2)
            NUM[f"e240_{arm}_{key}"] = dict(values=v, mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else None)
        for s in seeds:
            ax.plot([0, 1], [card(f"e24_j1eq_s{s}")[key], card(f"e24_j1rc_s{s}")[key]], color="#b8bec7", lw=0.8, zorder=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["j1eq", "j1rc"], fontsize=9)
        ax.set_xlim(-0.6, 1.6)
        style(ax, title, key)
    ax = axs[2]
    pts = [25, 50, 75, 100]
    for s in (2, 3):
        have = [card(f"e24_j1rc_s{s}_at{p}") and card(f"e24_j1eq_s{s}_at{p}") for p in pts[:3]] + [card(f"e24_j1rc_s{s}") and card(f"e24_j1eq_s{s}")]
        if not all(have):
            continue
        gap = []
        for p in pts:
            suf = f"_at{p}" if p < 100 else ""
            gap.append((card(f"e24_j1rc_s{s}{suf}")["lanes_found"] - card(f"e24_j1eq_s{s}{suf}")["lanes_found"], card(f"e24_j1rc_s{s}{suf}")["Q"] - card(f"e24_j1eq_s{s}{suf}")["Q"]))
        NUM[f"e241_seed{s}"] = dict(points=pts, lanes_gap=[g[0] for g in gap], q_gap=[g[1] for g in gap])
        ax.plot(pts, [g[0] for g in gap], marker="o", color=C["rc"], label=f"lanes found gap, seed {s}", ls="-" if s == 2 else "--")
        ax.plot(pts, [g[1] for g in gap], marker="s", color=C["eq"], label=f"Q gap, seed {s}", ls="-" if s == 2 else "--")
    ax.axhline(0, color="#444444", lw=0.6)
    ax.set_xlabel("share of the 4 u fine-tune", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False)
    style(ax, "j1rc - j1eq along the fine-tune (E241)", "difference")
    fig.tight_layout()
    fig.savefig(f"{FIG}/w3_seed_floor.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- 2 matched vertex budget (E255 grids, E244 partial)
def fig_matched():
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
    ax = axs[0]
    for name, col, lab in (("e24_j1ex_s0", C["ex"], "j1eq on the exact map, seed 0"), ("e24_j1ex_s1", "#7bc48a", "j1eq on the exact map, seed 1"), ("e25_j1phd2_s0", C["phd"], "Poisson set loss, full schedule, seed 0")):
        p = f"{R}/runs/eval/{name}/m143_grid.json"
        if not os.path.exists(p):
            continue
        g = json.load(open(p))
        NUM[f"matched_{name}"] = dict(tau=g["tau"], vtx_r=g["vtx_r"])
        for tau in sorted({x["tau"] for x in g["grid"]}):
            rows = sorted([x for x in g["grid"] if x["tau"] == tau], key=lambda x: x["vtx_r"])
            ax.plot([x["vert100"] for x in rows], [x["Q"] for x in rows], marker=".", color=col, alpha=0.35 + 0.65 * (tau == g["tau"]), lw=1)
        ax.plot([], [], color=col, label=lab)
    ax.axvline(14.3, color="#444444", lw=0.8, ls="--")
    ax.set_xlabel("vertices per 100 m (val250, grid of tau x vtx_r)", fontsize=9)
    ax.set_ylim(0.6, 0.76)
    ax.legend(fontsize=7.5, frameon=False)
    style(ax, "Q against the vertex budget, tuning subset", "Q")
    ax = axs[1]
    pts = []
    for p in sorted(glob.glob(f"{R}/runs/eval/e24_j1phd_s0_at50/val_*.json")):
        m = json.load(open(p, encoding="utf-8-sig"))["metrics"]
        pts.append((m["vert100"], m["Q"], os.path.basename(p)[4:-5]))
    m0 = card("e24_j1phd_s0_at50")
    if m0:
        pts.append((m0["vert100"], m0["Q"], "tau0.5"))
    if pts:
        pts.sort()
        ax.scatter([a for a, _, _ in pts], [b for _, b, _ in pts], marker="o", s=30, color=C["phd"], label="Poisson loss, 50 % checkpoint (E244.a0 cut): tau and vtx_r sweeps")
        NUM["e244_at50_points"] = [dict(vert100=a, Q=b, tag=c) for a, b, c in pts]
    for name, col, lab in (("e24_j1eq_s2_at50", C["eq"], "j1eq seed 2, 50 %"), ("e24_j1rc_s2_at50", C["rc"], "j1rc seed 2, 50 %")):
        m = card(name)
        if m:
            ax.scatter([m["vert100"]], [m["Q"]], s=70, color=col, zorder=4, label=lab)
    ax.axvline(14.3, color="#444444", lw=0.8, ls="--")
    ax.set_xlabel("vertices per 100 m (VAL 873)", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False)
    style(ax, "E244 arm a at 50 % of the schedule (tau and vtx_r sweep on VAL)", "Q")
    fig.tight_layout()
    fig.savefig(f"{FIG}/w3_matched_vertex.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- 3 registration replicate (E238)
def fig_regis():
    p = f"{R}/runs/e17/tables/e238.md"
    if not os.path.exists(p):
        return
    txt = open(p, encoding="utf-8").read().split("\n\n")
    names = {"dark": "night", "blur": "blur", "cam_front3_off": "3 cameras off"}
    rows = {}
    for block in txt:
        lines = [l for l in block.strip().split("\n") if l.startswith("|")]
        if not lines:
            continue
        key = lines[0].split("|")[1].strip()
        for l in lines:
            c = [x.strip() for x in l.split("|")[1:-1]]
            m = re.match(r"boot (\w+) - (\w+)", c[0])
            if m and c[0].startswith("boot") and "fail" in c[1]:
                v = re.match(r"F1@1 ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]", c[3])
                rows.setdefault((key, c[0]), tuple(float(x) for x in v.groups()))
    NUM["e238_boot_blocks_seen"] = len({k for k, _ in rows})
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    labs = [("boot R2d1 - U2d1", "draw 1: registered - unregistered"), ("boot R2d2 - U2d2", "draw 2: registered - unregistered"), ("boot R2d0 - U2d0", "stored draw 0: registered - unregistered")]
    pert = [k for k in ("dark", "blur", "cam_front3_off")]
    w = 0.25
    for j, (lk, ll) in enumerate(labs):
        d, lo, hi = [], [], []
        for pn in pert:
            r = rows.get((pn, lk))
            d.append(r[0] if r else np.nan)
            lo.append(r[0] - r[1] if r else 0)
            hi.append(r[2] - r[0] if r else 0)
        ax.bar(np.arange(3) + (j - 1) * w, d, w, yerr=[lo, hi], color=["#1f5fa8", "#5b8fd1", "#9fbbe3"][j], label=ll, capsize=2)
        NUM[f"e238_{lk}"] = {pn: rows.get((pn, lk)) for pn in pert}
    ax.axhline(0, color="#444444", lw=0.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels([names[p] for p in pert], fontsize=9)
    ax.legend(fontsize=7.5, frameon=False)
    style(ax, "E238: F1@1 gain of registration + trust on the failure half (95 % scene intervals)", "F1@1 difference")
    fig.tight_layout()
    fig.savefig(f"{FIG}/w3_registration.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- 4 metric by city (E252)
def fig_city():
    p = glob.glob(f"{R}/experiments/E252_*/gt_offset.txt")
    if not p:
        return
    txt = open(p[0], encoding="utf-8").read()
    models = re.split(r"\n== ", "\n" + txt)[1:]
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.8))
    res = {}
    for blk in models:
        nm = blk.split("\n")[0].strip()
        res[nm] = {}
        for city in ("boston-seaport", "singapore-onenorth"):
            m = re.search(rf"^{city}\s+(\d+)\s+(\d+)\s+([\d.]+)\s+\[([\d.]+),([\d.]+)\]\s+([\d.]+)\s+([\d.]+)", blk, re.M)
            if m:
                res[nm][city] = dict(frames=int(m.group(1)), scenes=int(m.group(2)), Q=float(m.group(3)), lo=float(m.group(4)), hi=float(m.group(5)), F1_05=float(m.group(6)), F1_10=float(m.group(7)))
        off = re.findall(r"^(boston-seaport|singapore-onenorth)\s+(\d+)\s+(\d+)\s+(-?[\d.]+) \[[^\]]*\]\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+) of (\d+)", blk, re.M)
        for city, ns, npnt, med, mabs, s05, s10, nb, nt in off:
            res[nm].setdefault(city, {}).update(median_abs=float(mabs), share05=float(s05), share10=float(s10))
    NUM["e252"] = res
    names = list(res)
    x = np.arange(len(names))
    ax = axs[0]
    for j, (city, col, lab) in enumerate((("boston-seaport", C["bos"], "Boston (10 scenes)"), ("singapore-onenorth", C["sg"], "Singapore One-North (12 scenes)"))):
        q = [res[n][city]["Q"] for n in names]
        lo = [res[n][city]["Q"] - res[n][city]["lo"] for n in names]
        hi = [res[n][city]["hi"] - res[n][city]["Q"] for n in names]
        ax.bar(x + (j - 0.5) * 0.36, q, 0.36, yerr=[lo, hi], color=col, label=lab, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0.6, 0.85)
    ax.legend(fontsize=7.5, frameon=False)
    style(ax, "E252: Q by city (VAL 873, 95 % scene intervals)", "Q")
    ax = axs[1]
    for j, (city, col) in enumerate((("boston-seaport", C["bos"]), ("singapore-onenorth", C["sg"]))):
        v = [res[n][city].get("share05", np.nan) for n in names]
        ax.bar(x + (j - 0.5) * 0.36, v, 0.36, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    style(ax, "share of correct points more than 0.5 m off the ground-truth line", "share")
    fig.tight_layout()
    fig.savefig(f"{FIG}/w3_city.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- 5 timing (E243 invalid, E253)
def fig_timing():
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, d, title in ((axs[0], "e243", "E243 (22:00, loop 19): INVALID session"), (axs[1], "e253", "E253 (loop 20)")):
        s = f"{R}/runs/e19/{d}/summary.json"
        if not os.path.exists(s):
            ax.text(0.5, 0.5, "not run", ha="center", va="center", transform=ax.transAxes)
            style(ax, title)
            continue
        j = json.load(open(s))
        arms = j["arms"]
        names = dict(a="j1\n192/96", c="f1\n288/144", b="j1eq", d="j1rc", e="rc1\n288/144", f="f1 + time\nrecursion")
        for i, a in enumerate("acbdef"):
            if a in arms:
                ax.scatter(np.full(len(arms[a]["replicates"]), i) + np.linspace(-0.15, 0.15, len(arms[a]["replicates"])), arms[a]["replicates"], s=22, color="#1f5fa8")
                ax.plot([i - 0.3, i + 0.3], [arms[a]["median"]] * 2, color="#c85f00", lw=2)
        NUM[d] = {a: dict(median=arms[a]["median"], replicates=arms[a]["replicates"]) for a in arms}
        NUM[d + "_verdict"] = j.get("verdict")
        ax.set_xticks(range(6))
        ax.set_xticklabels([names[a] for a in "acbdef"], fontsize=8)
        style(ax, title, "ms per frame" if d == "e243" else None)
    fig.tight_layout()
    fig.savefig(f"{FIG}/w3_timing.png", dpi=130)
    plt.close(fig)


for f in (fig_seed_floor, fig_matched, fig_regis, fig_city, fig_timing):
    try:
        f()
        print("ok", f.__name__)
    except Exception as e:  # a figure of a run that did not finish must not stop the others
        print("FAILED", f.__name__, repr(e))
json.dump(NUM, open(f"{R}/reports/source/weekly/week3_numbers.json", "w"), indent=1, default=float)
print("numbers written")
