"""Bericht E56 (Schleife 4): Richtungs-Konsistenz als Annahmekriterium der Bezier-Verbinder (E53).
Liest out/loop2/e56_h<tau_h>_c<cos_c>/assemble.json (VAL-Durchlauf), e53_t* / e53_f* (ohne Kriterium), e44_cons_p (Standard), e56_mid (VAL + TEST der vorab festgelegten mittleren Einstellung); Text loop2/loop4_text_e56.py (TEXT56).  Aufruf (gcar): make_e56.py [figs|report]"""
import glob
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch_figs4 as AF4  # noqa: E402
from report2 import f3, make_report2  # noqa: E402

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
FIG = f"{OUT}/figs"
_ns = {}
if os.path.exists("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop4_text_e56.py"):
    exec(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop4_text_e56.py", encoding="utf-8").read(), _ns)
TEXT = _ns.get("TEXT56", {})


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def score(m):
    return m["split6_F1"] + m["merge6_F1"] + 2.0 * m["route_f"] + 2.0 * m["F1@1.0"]


def main(mode):
    head = ["variant", "set", "F1@1m", "P / R @1m", "route rec / prec / F", "split F1 3 / 6 m", "split P / R @6m", "merge F1 3 / 6 m", "merge P / R @6m", "dir acc", "VAL selection score", "path F1"]
    paths = jl(f"{OUT}/e46/path_eval_all.json") or {}
    rows = []

    def add(lab, st, m, pk=None):
        pf = paths.get(f"{pk}_{st.lower()}", {}).get("path_f1") if pk else None
        rows.append([lab, st, f3(m["F1@1.0"]), f"{f3(m['P@1.0'])} / {f3(m['R@1.0'])}", f"{f3(m['route_rec'])} / {f3(m['route_prec'])} / {f3(m['route_f'])}", f"{f3(m['split_F1'])} / {f3(m['split6_F1'])}", f"{f3(m['split6_P'])} / {f3(m['split6_R'])}", f"{f3(m['merge_F1'])} / {f3(m['merge6_F1'])}", f"{f3(m['merge6_P'])} / {f3(m['merge6_R'])}", f3(m.get("dir_acc")), f"{score(m):.3f}" if st == "VAL" else "-", f3(pf)])

    d0 = jl(f"{OUT}/e44_cons_p/assemble.json")
    if d0:
        for st in ("val", "test"):
            add("previous final system (no key points)", st.upper(), d0["base"][st], "e43/graphs_rule_e41_A")
        for st in ("val", "test"):
            add("DEFAULT key-point nodes (E44)", st.upper(), d0[st], "e44_cons_p/graphs")
    sweep = {}
    for th in ("0.10", "0.50"):
        d = jl(f"{OUT}/e53_t{th}/assemble.json") or jl(f"{OUT}/e53_f{th}/assemble.json")
        if d:
            add(f"connectors, heat >= {th}, no direction criterion (E53)", "VAL", d["val"], f"e53_f{th}/graphs")
            sweep.setdefault(th, {})["none"] = d["val"]
    found = []
    for fn in sorted(glob.glob(f"{OUT}/e56_h*_c*/assemble.json")):
        mm = re.search(r"e56_h([\d.]+)_c([\d.]+)/", fn)
        d = jl(fn)
        if mm and d and "val" in d:
            found.append((mm.group(1), mm.group(2), d["val"]))
    for th, cc, m in found:
        add(f"connectors, heat >= {th}, direction cosine >= {cc}", "VAL", m)
        sweep.setdefault(th, {})[cc] = m
    sel = jl(f"{OUT}/e56_mid/assemble.json")
    if sel and sel.get("best"):
        pr = sel["best"]["par"]
        for st in ("val", "test"):
            if st in sel:
                add(f"PRE-SET middle setting: heat >= {pr.get('synth')}, direction cosine >= {pr.get('cos_c')} (VAL + TEST)", st.upper(), sel[st], "e56_mid/graphs")
    images = []
    if sweep:
        fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
        for th, col in (("0.10", "#1f77b4"), ("0.50", "#d62728")):
            if th not in sweep:
                continue
            xs, ys = [], []
            for cc in ("none", "0.3", "0.6", "0.8"):
                if cc in sweep[th]:
                    xs.append(-0.3 if cc == "none" else float(cc))
                    ys.append(sweep[th][cc])
            for ax, key, lab in ((axs[0], "split6_F1", "split F1 @6 m (VAL)"), (axs[1], "merge6_F1", "merge F1 @6 m (VAL)"), (axs[2], "route_f", "route F (VAL)")):
                ax.plot(xs, [y[key] for y in ys], "-o", color=col, label=f"heat >= {th}")
                ax.set_title(lab, fontsize=9)
                ax.set_xlabel("min. direction cosine (-0.3 = off)", fontsize=8)
                ax.grid(alpha=0.3)
        if d0:
            for ax, key in ((axs[0], "split6_F1"), (axs[1], "merge6_F1"), (axs[2], "route_f")):
                ax.axhline(d0["val"][key], color="k", ls=":", label="default (no connectors)")
        axs[0].legend(fontsize=7)
        fig.tight_layout()
        p = f"{FIG}/e56_sweep.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        images.append((p, "Direction-consistency criterion for the connectors (VAL): split F1, merge F1 and route F against the minimum cosine between the curve tangent and the predicted direction field; left-most point: criterion off (E53)."))
    if mode == "figs":
        for r in rows:
            print(r)
        return
    t = TEXT
    make_report2("E56", t.get("title", "E56"), t.get("question", ""), t.get("setup", []), [("Connector acceptance with the predicted travel direction as an additional criterion", head, rows)], images, t.get("findings", ["(to be written)"]), t.get("limits", []), arch=AF4.dirconn_arch)
    print("E56 written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figs")
