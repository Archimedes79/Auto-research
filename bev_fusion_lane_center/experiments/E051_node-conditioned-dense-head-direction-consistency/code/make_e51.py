"""Bericht E51 (Schleife 4): Knoten-konditionierter dichter Kopf (SAM-Road: Schluesselpunkte -> Strassenmaske). Aufruf (gcar): make_e51.py [figs|report]
Liest out/loop2/e41_A, e51_A_nodes, e51_A_nodes_oracle (metrics.json), e51/metrics_e51_A_nodes_noprop.json, e43/metrics_e41_A_noprop.json, e51_cons/assemble.json, e44_cons_p/assemble.json; Text loop2/loop4_text_e51.py (TEXT51)."""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch_figs4 as AF4  # noqa: E402
import dense_report_lib as DR  # noqa: E402
from report2 import f3, make_report2  # noqa: E402

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
FIG = f"{OUT}/figs"
_ns = {}
if os.path.exists("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop4_text_e51.py"):
    exec(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop4_text_e51.py", encoding="utf-8").read(), _ns)
TEXT = _ns.get("TEXT51", {})


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def main(mode):
    head = ["dense head (temporal A, fine-tuned 2,000 steps)", "TRAIN_EVAL F1@1m", "VAL F1@1m", "VAL P / R @1m", "TEST F1@1m", "TEST P / R @1m", "VAL recall @1m", "TEST recall @1m"]
    rows, runs = [], {}
    for nm, lab in (("e41_A", "control: temporal A, no node input (E41)"), ("e51_A_nodes", "+ node heat maps as input (predicted key points at test)"), ("e51_A_nodes_oracle", "same model with GROUND-TRUTH node maps at test (upper bound)")):
        r = DR.load(nm)
        if not r or "val" not in r:
            continue
        bt = DR.best_tau(r)
        runs[nm] = (r, lab)
        v, t = r["val"][bt], r["test"][bt]
        a = r["train_eval"][bt]["F1@1.0"] if "train_eval" in r else None
        rows.append([lab, f3(a) if a is not None else "-", f3(v["F1@1.0"]), f"{f3(v['P@1.0'])} / {f3(v['R@1.0'])}", f3(t["F1@1.0"]), f"{f3(t['P@1.0'])} / {f3(t['R@1.0'])}", f3(v["R@1.0"]), f3(t["R@1.0"])])
    h2 = ["system (single frame + temporal A; refiner + rule links)", "set", "F1@1m", "P / R @1m", "route F", "split F1 3 / 6 m", "merge F1 3 / 6 m", "dir acc"]
    r2 = []

    def add(lab, st, m):
        r2.append([lab, st, f3(m["F1@1.0"]), f"{f3(m['P@1.0'])} / {f3(m['R@1.0'])}", f3(m["route_f"]), f"{f3(m['split_F1'])} / {f3(m['split6_F1'])}", f"{f3(m['merge_F1'])} / {f3(m['merge6_F1'])}", f3(m.get("dir_acc"))])

    a41 = jl(f"{OUT}/e43/metrics_e41_A_noprop.json")
    a51 = jl(f"{OUT}/e51/metrics_e51_A_nodes_noprop.json")
    c41 = jl(f"{OUT}/e44_cons_p/assemble.json")
    c51 = jl(f"{OUT}/e51_cons/assemble.json")
    for st in ("val", "test"):
        if a41 and st in a41:
            add("A (E41): dense head with temporal fusion", st.upper(), a41[st]["rule"]["metrics"])
        if c41 and st in c41:
            add("A + key-point nodes (conservative cut + prune)", st.upper(), c41[st])
        if a51 and st in a51:
            add("E51: node-conditioned dense head", st.upper(), a51[st]["rule"]["metrics"])
        if c51 and st in c51:
            add("E51 + key-point nodes (conservative cut + prune)", st.upper(), c51[st])
    images = []
    if len(runs) >= 2:
        fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
        DR.plot_pr(axs[0], {k: v[0] for k, v in runs.items()}, "val", labels={k: v[1][:46] for k, v in runs.items()})
        DR.plot_pr(axs[1], {k: v[0] for k, v in runs.items()}, "test", labels={k: v[1][:46] for k, v in runs.items()})
        fig.tight_layout()
        p = f"{FIG}/e51_pr.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        images.append((p, "Precision-recall of the dense heads (x = VAL-chosen threshold)."))
    if mode == "figs":
        for r in rows + r2:
            print(r)
        return
    t = TEXT
    make_report2("E51", t.get("title", "E51"), t.get("question", ""), t.get("setup", []), [("Dense level", head, rows), ("System level (proposals + refiner e39_main + rule links; the key-point head of E44 supplies the node maps)", h2, r2)], images, t.get("findings", ["(to be written)"]), t.get("limits", []), arch=AF4.nodecond_arch)
    print("E51 written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figs")
