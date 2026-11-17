"""E375: the per-class cell by the rule of the card (section 0, v2), from the two per-class TRAIN tables of the E367 b_bal25 heads; plus the diagonal consistency check against the stored 24-cell tables (scratch)"""
import json
import os

runs = ["e367_b25_s0", "e367_b25_s1"]
pc = [json.load(open(os.path.join("runs", r, "token_sweep_perclass_train_every3.json"))) for r in runs]
old = [json.load(open(os.path.join("runs", r, "token_sweep_train_every3.json"))) for r in runs]
key = lambda r: (r["link_thr"], r.get("link_thr_long", r["link_thr"]), r["keep_thr"], r["none_gate"])  # noqa: E731

# diagonal check: per-class cells with short = long must equal the stored gate-off cells
bad = 0
for t_pc, t_old in zip(pc, old):
    om = {(r["link_thr"], r["keep_thr"], r["none_gate"]): r for r in t_old["rows"]}
    for r in t_pc["rows"]:
        if r["link_thr"] == r.get("link_thr_long", r["link_thr"]):
            o = om.get((r["link_thr"], r["keep_thr"], r["none_gate"]))
            if o is None:
                continue
            for k in ("topo_F1", "lanes_found", "dup_frac", "F1@1.0"):
                if abs(o[k] - r[k]) > 1e-9:
                    bad += 1
                    print("DIAGONAL MISMATCH", key(r), k, o[k], r[k])
print("diagonal check:", "PASS" if bad == 0 else f"FAIL ({bad})")

cells = {}
for t in pc:
    for r in t["rows"]:
        cells.setdefault(key(r), []).append(r)
assert all(len(v) == 2 for v in cells.values())
mean = lambda v, k: sum(r[k] for r in v) / len(v)  # noqa: E731
champ = (0.2, 0.2, 0.6, False)
lanes_champ = mean(cells[champ], "lanes_found")
rows = []
for c, v in cells.items():
    rows.append(dict(cell=c, topo=mean(v, "topo_F1"), dup=mean(v, "dup_frac"), lanes=mean(v, "lanes_found"), junc=mean(v, "junction_F1"),
                     jP=mean(v, "junction_P"), jR=mean(v, "junction_R"), f1=mean(v, "F1@1.0"), jpts=mean(v, "junction_points_per_frame")))
rows.sort(key=lambda x: -x["topo"])
print("champion cell", champ, "mean lanes found", round(lanes_champ, 4))
print("cell (short, long, keep, gate) | topo | junction F1 (P / R) | junction points / frame | lanes | dup | F1@1.0")
for r in rows:
    print(r["cell"], round(r["topo"], 4), round(r["junc"], 4), f"({r['jP']:.3f} / {r['jR']:.3f})", round(r["jpts"], 2), round(r["lanes"], 4), round(r["dup"], 4), round(r["f1"], 4), "<- champion" if r["cell"] == champ else "")
ok = [r for r in rows if r["dup"] <= 0.05 and r["lanes"] >= lanes_champ - 0.01]
best = max(r["topo"] for r in ok)
win = [r for r in ok if r["topo"] >= best - 0.005]


def dist(c):
    return (abs(c[0] - champ[0]), abs(c[1] - champ[1]), abs(c[2] - champ[2]), -c[0], -c[1], -c[2])


win.sort(key=lambda r: (-r["junc"], dist(r["cell"])))
print("eligible", len(ok), "| best topo", round(best, 4), "| window", len(win))
print("CHOSEN", win[0]["cell"], {k: round(win[0][k], 4) for k in ("topo", "junc", "jP", "jR", "lanes", "dup", "f1")})
