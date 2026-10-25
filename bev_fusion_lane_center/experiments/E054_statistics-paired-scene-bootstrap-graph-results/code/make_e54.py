"""Bericht E54 (Schleife 4): gepaarter Szenen-Bootstrap der Kopfzahlen.  Liest out/loop2/e54_boot/boot.json (boot_eval.py); Text loop2/loop4_text_e54.py (TEXT54).  Aufruf (gcar): make_e54.py [figs|report]"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch_figs4 as AF4  # noqa: E402
from report2 import make_report2  # noqa: E402

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
FIG = f"{OUT}/figs"
_ns = {}
if os.path.exists("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop4_text_e54.py"):
    exec(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop4_text_e54.py", encoding="utf-8").read(), _ns)
TEXT = _ns.get("TEXT54", {})

LAB = dict(base="previous final system (loop 3, no key points)", default="+ key-point nodes, DEFAULT (E44)", loose="+ key-point nodes, node priority (E44)", conn50="+ Bezier connectors, heat >= 0.5 (E53)", conn60="+ Bezier connectors, heat >= 0.6 (E53)", conn50dir="+ Bezier connectors, heat >= 0.5 + direction cosine >= 0.6 (E56)", map_base="with map prior, exact pose (loop 3)", map_default="with map prior + key-point nodes, DEFAULT (E44)", map_conn50="with map prior + connectors, heat >= 0.5 (E53)", map_conn60="with map prior + connectors, heat >= 0.6 (E53)")
PAIR_LAB = {"default-base": "key-point nodes (DEFAULT) - previous system", "loose-base": "key-point nodes (node priority) - previous system", "conn50-default": "connectors 0.5 - DEFAULT", "conn50dir-conn50": "connectors 0.5 + direction criterion - connectors 0.5", "conn50dir-default": "connectors 0.5 + direction criterion - DEFAULT", "conn60-default": "connectors 0.6 - DEFAULT", "conn50-base": "connectors 0.5 - previous system", "conn60-base": "connectors 0.6 - previous system", "map_base-base": "map prior (exact pose) - no map", "map_default-map_base": "map prior + key-point nodes - map prior", "map_default-default": "map prior + key-point nodes - key-point nodes without map", "map_conn50-map_default": "map prior + connectors 0.5 - map prior + key-point nodes", "map_conn50-map_base": "map prior + connectors 0.5 - map prior (loop 3)", "map_conn60-map_default": "map prior + connectors 0.6 - map prior + key-point nodes", "map_conn50-conn50": "map prior + connectors 0.5 - connectors 0.5 without map"}
MET = [("F1@1m", "point F1 @1 m"), ("route_f", "route F"), ("split6_F1", "split F1 @6 m"), ("merge6_F1", "merge F1 @6 m")]


def main(mode):
    D = json.load(open(f"{OUT}/e54_boot/boot.json"))
    h1 = ["system", "set"] + [lab for _, lab in MET] + ["direction acc"]
    r1 = []
    for st in ("test", "val"):
        for k in ("base", "default", "loose", "conn50", "conn50dir", "conn60", "map_base", "map_default", "map_conn50", "map_conn60"):
            s = D[st]["systems"].get(k)
            if s:
                r1.append([LAB[k], st.upper()] + [f"{s[m]['value']:.3f} [{s[m]['lo']:.3f}, {s[m]['hi']:.3f}]" for m, _ in MET] + [f"{s['dir_acc']['value']:.3f}"])
    h2 = ["difference (A - B)", "set"] + [lab for _, lab in MET] + ["direction acc"]
    r2 = []
    for st in ("test", "val"):
        for k in ("default-base", "loose-base", "conn50-default", "conn60-default", "conn50dir-conn50", "conn50dir-default", "conn50-base", "conn60-base", "map_base-base", "map_default-map_base", "map_default-default", "map_conn50-map_default", "map_conn50-map_base", "map_conn60-map_default", "map_conn50-conn50"):
            p = D[st]["pairs"].get(k)
            if p:
                cell = lambda m: f"{p[m]['diff']:+.3f} [{p[m]['lo']:+.3f}, {p[m]['hi']:+.3f}]" + (" *" if (p[m]["lo"] >= 0.0005 or p[m]["hi"] <= -0.0005) else "")
                r2.append([PAIR_LAB[k], st.upper()] + [cell(m) for m, _ in MET] + [cell("dir_acc")])
    images = []
    keys = ["default-base", "conn50-default", "conn60-default", "conn50-base", "map_base-base", "map_default-map_base"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    for ax, (m, lab) in zip(axs, MET[1:]):
        for j, k in enumerate(keys):
            for st, c, off in (("test", "#d62728", -0.13), ("val", "#1f77b4", 0.13)):
                p = D[st]["pairs"].get(k)
                if not p:
                    continue
                ax.errorbar(p[m]["diff"], j + off, xerr=[[p[m]["diff"] - p[m]["lo"]], [p[m]["hi"] - p[m]["diff"]]], fmt="o", color=c, capsize=2.5, label=st.upper() if j == 0 else None, ms=4)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"change of {lab}", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels([PAIR_LAB[k] for k in keys], fontsize=7.5)
        ax.invert_yaxis()
    axs[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    p = f"{FIG}/e54_forest.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    images.append((p, "Paired differences with 95 % scene-bootstrap intervals (red: TEST, 21 scenes; blue: VAL, 22 scenes); an interval that does not contain 0 is a difference larger than the scene-to-scene variation of the evaluation set."))
    if mode == "figs":
        for r in r1 + r2:
            print(r)
        return
    t = TEXT
    make_report2("E54", t.get("title", "E54"), t.get("question", ""), t.get("setup", []), [("Scores with 95 % scene-bootstrap intervals [low, high]", h1, r1), ("Paired differences A - B (same bootstrap draws; * = the interval excludes 0 by at least 0.0005)", h2, r2)], images, t.get("findings", ["(to be written)"]), t.get("limits", []), arch=AF4.boot_arch)
    print("E54 written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figs")
