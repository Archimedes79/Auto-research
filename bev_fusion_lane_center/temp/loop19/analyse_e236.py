"""E236 analysis: Q against vertices per 100 m along the tau sweep (fork pruning 0.4), the tau per model that hits 14.3 +- 0.15, the matched table.

    wsl python3 temp/loop19/analyse_e236.py pick      # prints 'PICK <model> <tau>' lines (and the curves) from runs/eval/<model>/val_sw<tau>.json
    wsl python3 temp/loop19/analyse_e236.py table     # after the matched evaluations (tag v143): the matched table, edge a - c, the decision clause

Pick rule (declared in the card): the sweep tau whose vert100 is nearest to 14.3 and inside +-0.15; if none, the nearest sweep point is used and marked
UNMATCHED (the next step is a finer tau; the sweep is tuning on VAL for both arms equally, no TEST).
"""
import glob
import json
import os
import sys

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "runs", "eval")
MODELS = ["champion", "recall", "recall_s1", "fast", "fast_recall"]
TARGET, TOL = 14.3, 0.15


def card(m, tag):
    p = f"{R}/{m}/val_{tag}.json"
    return json.load(open(p))["metrics"] if os.path.exists(p) else None


def sweep(m, kind="vx"):
    out = []
    for p in sorted(glob.glob(f"{R}/{m}/val_{kind}*.json")):
        tau = float(os.path.basename(p)[len("val_" + kind):-len(".json")])
        out.append((tau, json.load(open(p))["metrics"]))
    return sorted(out)


def pick():
    for m in MODELS:
        s = sweep(m)
        if not s:
            print(f"# {m}: no sweep cards")
            continue
        print(f"# {m}: " + "  ".join(f"v{t:.1f}:{x['vert100']:.2f}/Q{x['Q']:.4f}" for t, x in s))
        t, x = min(s, key=lambda a: abs(a[1]["vert100"] - TARGET))
        print(f"PICK {m} {t:.2f}   # vert100 {x['vert100']:.2f} Q {x['Q']:.4f} lanes_found {x['lanes_found']:.3f} " + ("matched" if abs(x["vert100"] - TARGET) <= TOL else "UNMATCHED"))


def table():
    rows = {m: card(m, "v143") for m in MODELS}
    print(f"{'model':12} {'vert100':>8} {'Q':>8} {'lanes_found':>11} {'topo_F1':>8} {'R@1_far':>8} {'J':>7} {'S':>7}")
    for m, x in rows.items():
        if x:
            print(f"{m:12} {x['vert100']:8.2f} {x['Q']:8.4f} {x['lanes_found']:11.3f} {x['topo_F1']:8.3f} {x['R@1_far']:8.3f} {x['J']:7.3f} {x['S']:7.3f}")
    reg = {"champion": card("champion", "val"), "recall": card("recall", "val"), "champion_noprune": card("champion_noprune", "val")}
    for m, x in reg.items():
        if x:
            print(f"{m + ' (registered)':26} vert100 {x['vert100']:.2f} Q {x['Q']:.4f} lanes_found {x['lanes_found']:.3f}")
    a, c, s1 = rows["recall"], rows["champion"], rows["recall_s1"]
    if a and c:
        dq, dl = a["Q"] - c["Q"], a["lanes_found"] - c["lanes_found"]
        verdict = "KEEP" if dq >= 0.010 and dl >= 0.04 else "CUT" if dq <= -0.005 else "TIE"
        print(f"\nQ(recall) - Q(champion) at matched vertices = {dq:+.4f}; lanes_found {dl:+.3f} -> {verdict} (the tool band is 0.010; the card's 0.005 clause is raised to it)")
        if s1:
            print(f"seed twin recall_s1: Q {s1['Q']:.4f} (edge {s1['Q'] - c['Q']:+.4f}), vert100 {s1['vert100']:.2f}")
        for m in ("recall", "champion"):
            print(f"{m}: Q at the sweep points nearest 14.3: " + ", ".join(f"{x['vert100']:.2f}->{x['Q']:.4f}" for t, x in sweep(m) if 13.5 <= x["vert100"] <= 15.5))


if __name__ == "__main__":
    {"pick": pick, "table": table}[sys.argv[1]]()
