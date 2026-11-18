"""Bericht E66 (Schleife 5): gelernter Planer auf der Modulausgabe.  Liest out/loop2/e66_planner/{metrics.json, examples.npz, data_test.npz} (planner_stub.py); Text loop2/loop5_text_e66.py (TEXT66).  Aufruf (gcar): make_e66.py [figs|report]"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arch_figs import blocks_arch  # noqa: E402
from report2 import f3, make_report2  # noqa: E402

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
FIG = f"{OUT}/figs"
EDIR = f"{OUT}/e66_planner"
os.makedirs(FIG, exist_ok=True)
_ns = {}
if os.path.exists("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop5_text_e66.py"):
    exec(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop5_text_e66.py", encoding="utf-8").read(), _ns)
TEXT = _ns.get("TEXT66", {})

ROWS = [
    ("const", "given_true", "straight at the current speed (no module output)"),
    ("P0_lanes_only", "clean", "P0: lanes only (heat map of the lanes + speed)"),
    ("P1_lanes_route", "clean", "P1: lanes + route heat map, exact map"),
    ("P1_lanes_route", "s1", "P1: map with 1 m pose noise"),
    ("P1_lanes_route", "none", "P1: no map (route heat map = the head's guess from the image)"),
    ("P1_lanes_route", "oracle", "P1: ORACLE route heat map (ground-truth route)"),
    ("P1_lanes_route", "wrong", "P1: WRONG route (the route of another sample), compliance with the GIVEN route"),
    ("P1_lanes_route", "wrong_vs_true_route", "P1: WRONG route, compliance with the TRUE route"),
]


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def table():
    m = jl(f"{EDIR}/metrics.json")
    rows = []
    if not m:
        return rows
    for nm, cond, lab in ROWS:
        d = m.get(nm, {}).get(cond)
        if not d:
            continue
        r = [lab]
        for grp in ("all", "curve", "straight"):
            r += [f3(d[grp]["l2_3s"], 2), f3(d[grp]["lat"], 2)]
        r += [f3(d["curve"]["dev"]), f3(d["curve"]["compliance"]), f3(d["all"]["compliance"])]
        rows.append(r)
    return rows


def examples(n=4):
    f = f"{EDIR}/examples.npz"
    if not os.path.exists(f):
        return None
    e = np.load(f)
    te = np.load(f"{EDIR}/data_test.npz", allow_pickle=True)
    curve = e["curve"]
    top = np.argsort(-np.abs(e["y"][:, 2, 0]))[: n * 4]  # die Bilder mit der groessten seitlichen Verschiebung nach 3 s (staerkste Kurven)
    sel = top[:: 4][:n]
    fig, axs = plt.subplots(2, len(sel), figsize=(3.3 * len(sel), 7.0), squeeze=False)
    for c, k in enumerate(sel):
        for r, (cond, lab) in enumerate((("clean", "exact route"), ("wrong", "WRONG route given"))):
            ax = axs[r, c]
            rg = te["route_clean" if cond == "clean" else "route_wrong"][k].astype(np.float32)
            ax.imshow(rg.T, origin="lower", extent=[-50, 50, -50, 50], cmap="viridis", vmin=0, vmax=1, alpha=0.9)
            y = e["y"][k]
            ax.plot(np.r_[0, y[:, 0]], np.r_[0, y[:, 1]], "o-", color="white", ms=4, lw=1.5, label="driven (1 / 2 / 3 s)")
            ax.plot(np.r_[0, e["P1_lanes_route_" + cond][k][:, 0]], np.r_[0, e["P1_lanes_route_" + cond][k][:, 1]], "s-", color="#ff7f0e", ms=4, lw=1.5, label="P1 (lanes + route)")
            ax.plot(np.r_[0, e["P0_lanes_only_clean"][k][:, 0]], np.r_[0, e["P0_lanes_only_clean"][k][:, 1]], "^-", color="#d62728", ms=4, lw=1.2, label="P0 (lanes only)")
            ax.plot(0, 0, "k^", ms=7)
            ax.set_xlim(-30, 30)
            ax.set_ylim(-10, 50)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(lab, fontsize=8)
            if r == 0 and c == 0:
                ax.legend(fontsize=6, loc="lower left")
    fig.suptitle("E66: TEST curve / junction frames - the route heat map given to the planner (colour), the driven positions (white) and the planned positions", fontsize=8)
    fig.tight_layout()
    p = f"{FIG}/e66_examples.png"
    fig.savefig(p, dpi=95)
    plt.close(fig)
    return p


def bars():
    m = jl(f"{EDIR}/metrics.json")
    if not m:
        return None
    labs = [("const", "given_true", "straight"), ("P0_lanes_only", "clean", "P0 lanes only"), ("P1_lanes_route", "clean", "P1 exact map"), ("P1_lanes_route", "s1", "P1 1 m noise"), ("P1_lanes_route", "none", "P1 no map"), ("P1_lanes_route", "wrong", "P1 WRONG route"), ("P1_lanes_route", "oracle", "P1 oracle route")]
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.7))
    for ax, key, ttl in ((axs[0], "l2_3s", "L2 error after 3 s (m), curve / junction frames"), (axs[1], "compliance", "end point within 1.5 m of the GIVEN route, curve / junction frames")):
        vals = [m[a][b]["curve"][key] if b in m.get(a, {}) else np.nan for a, b, _ in labs]
        ax.bar(range(len(labs)), vals, color=["#7f7f7f", "#d62728", "#1f77b4", "#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c"])
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels([l for _, _, l in labs], fontsize=7, rotation=15)
        ax.set_title(ttl, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = f"{FIG}/e66_bars.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def arch(ax):
    b = [
        (0.1, 1.95, 2.5, 0.85, "frozen features + map +\nnavigation route\n(VAL, held-out outputs)", "in"),
        (0.1, 0.35, 2.5, 0.85, "ego speed v0", "data"),
        (3.2, 1.95, 2.6, 0.85, "route-conditioned head\n(E65d): lane heat map +\nROUTE heat map", "net"),
        (6.4, 1.15, 2.6, 1.25, "small CNN planner\n(4 conv + MLP, 0.3 M)\nP0: lanes only / P1: + route", "net"),
        (9.6, 1.15, 2.4, 1.25, "positions after\n1 / 2 / 3 s\n(ego frame)", "ok"),
    ]
    e = [(0, 2), (2, 3), (1, 3), (3, 4)]
    blocks_arch(ax, b, e, 12.4, 3.0)
    ax.set_title("E66: a learned planner on the module output (lane + route heat maps) - does it use the route it is given?", fontsize=10, loc="left")


def main(mode):
    head = ["planner / route given", "all: L2 3 s (m)", "all: lateral (m)", "curve: L2 3 s (m)", "curve: lateral (m)", "straight: L2 3 s (m)", "straight: lateral (m)", "curve: share > 2 m", "curve: end point on the given route", "all: end point on the given route"]
    if mode == "figs":
        for r in table():
            print(r)
        return
    images = []
    for fn, cap in ((bars, "Learned planner on the module output (TEST, mean of 3 seeds): L2 error after 3 s and compliance with the given route on curve / junction frames."), (examples, "Planned positions of P1 and P0 against the driven ones on the strongest curves of TEST: the effect of the route is a small lateral shift towards it; with a WRONG route given (bottom row) the planned positions hardly change - the planner uses the route as a scene cue and does not follow it.")):
        try:
            p = fn()
            if p:
                images.append((p, cap))
        except Exception as ex:  # noqa: BLE001
            print(fn.__name__, "Fehler:", ex)
    t = TEXT
    m = jl(f"{EDIR}/metrics.json") or {}
    cap = f"Learned planner (mean of {m.get('args', {}).get('seeds', 3)} training seeds), TEST: {m.get('n_test', '?')} frames, {m.get('n_curve', '?')} on curve / junction"
    make_report2("E66", t.get("title", "E66"), t.get("question", ""), t.get("setup", []), [(cap, head, table())], images, t.get("findings", ["(to be written)"]), t.get("limits", []), arch=arch, arch_height=3.0)
    print("E66 written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figs")
