import json
import os
import sys

arms = [a for a in ("p0", "p1", "p2", "q2") if os.path.exists(f"runs/e287_{a}_s0/ends_card.json")]
if len(sys.argv) > 1:
    arms = sys.argv[1:]


def load(a, s):
    h, c = f"runs/e287_{a}_s{s}/ends_head.json", f"runs/e287_{a}_s{s}/ends_card.json"
    return (json.load(open(h)) if os.path.exists(h) else None, json.load(open(c)) if os.path.exists(c) else None)


rows = []
for a in arms:
    for s in (0, 1):
        h, c = load(a, s)
        if not h or not c:
            rows.append(f"| **{a}** s{s} | pending |")
            continue
        n, d = c["numbers"]["head"], c["difference"]
        rows.append(f"| **{a}** s{s} | {h['ent_recall']:.3f} / {h['ext_recall']:.3f} | {h['pair_recall']:.3f} / {h['pair_precision']:.3f} | {n['topo_F1']:.4f} ({d['topo_F1']:+.4f}) | {n['F1@1.0']:.4f} ({d['F1@1.0']:+.4f}) | {n['lanes_found']:.4f} ({d['lanes_found']:+.4f}) | {n['junction_F1']:.4f} ({d['junction_F1']:+.4f}) | {n['junction_R']:.4f} ({d['junction_R']:+.4f}) | {n['dup_frac']:.4f} ({d['dup_frac']:+.4f}) | {n['vert100_dp']:.2f} ({d['vert100_dp']:+.2f}) |")
head = "| arm | entry / exit recall | pair recall / precision (logit >= 0) | topo F1 (vs plain) | F1@1.0 | lanes found | junction F1 | junction recall | dup_frac | vertices / 100 m |\n|---|---|---|---|---|---|---|---|---|---|"
print(head)
print("\n".join(rows))
h, c = load(arms[0], 0)
if c:
    b = c["numbers"].get("base", {})
    print("\nplain graph (control, seed 0): topo F1 %.4f, F1@1.0 %.4f, lanes found %.4f, junction F1 %.4f, junction recall %.4f, dup_frac %.4f, vertices %.2f" % (b.get("topo_F1", float("nan")), b.get("F1@1.0", float("nan")), b.get("lanes_found", float("nan")), b.get("junction_F1", float("nan")), b.get("junction_R", float("nan")), b.get("dup_frac", float("nan")), b.get("vert100_dp", float("nan"))))
