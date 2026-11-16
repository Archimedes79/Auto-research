"""E291 numbers table from the cards (plain python): snapped topo F1 (head) of the twin, of b's read-outs and of the stored E312, per seed; plus lanes found / junction F1 / dup_frac."""
import json
import os

ROOT = os.getcwd()


def card(run, name):
    d = json.load(open(os.path.join(ROOT, "runs", run, name)))
    n = d["numbers"]
    return n["head"], n.get("base")


rows = []
for S in (0, 1):
    out = {}
    h, b = card(f"e312_p_s{S}", "ends_card_snap_thr0.1.json")
    out["E312 stored, snapped"] = h
    out["B plain graph (base)"] = b
    h, _ = card(f"e312_p_s{S}", "ends_card_thr0.1.json")
    out["E312 stored, unsnapped"] = h
    h, _ = card(f"e291_c_s{S}", "ends_card_snap_thr0.1.json")
    out["twin c, snapped"] = h
    h, _ = card(f"e291_c_s{S}", "ends_card_thr0.1.json")
    out["twin c, unsnapped"] = h
    for r in ("clean", "noise10", "noise30", "pm1", "u", "s", "sc"):
        h, _ = card(f"e291_b_s{S}_{r}", f"ends_card_{r}_snap_thr0.1.json")
        out[f"b {r}, snapped"] = h
    h, _ = card(f"e291_b_s{S}_clean", "ends_card_clean_thr0.1.json")
    out["b clean, unsnapped"] = h
    rows.append(out)
keys = list(rows[0])
print("| read-out (VAL 873) | topo F1 s0 / s1 | lanes found s0 / s1 | junction F1 s0 / s1 | dup_frac s0 / s1 |")
print("|---|---|---|---|---|")
for k in keys:
    a, b = rows[0][k], rows[1][k]
    print(f"| {k} | {a['topo_F1']:.4f} / {b['topo_F1']:.4f} | {a['lanes_found']:.4f} / {b['lanes_found']:.4f} | {a['junction_F1']:.4f} / {b['junction_F1']:.4f} | {a['dup_frac']:.4f} / {b['dup_frac']:.4f} |")
print()
for S in (0, 1):
    t = rows[S]["twin c, snapped"]["topo_F1"]
    e = rows[S]["E312 stored, snapped"]["topo_F1"]
    print(f"seed {S}: twin minus stored E312 snapped {t - e:+.4f}; b clean minus stored {rows[S]['b clean, snapped']['topo_F1'] - e:+.4f}; b s minus stored {rows[S]['b s, snapped']['topo_F1'] - e:+.4f}")
