"""E357 T0: the verdict table by the written rule of the card (section 0). Reads runs/e357_p_s<S>/tok_card_thr0.5_sec0.json and the control cards."""
import json, os
def met(p):
    d = json.load(open(p)); return d.get("metrics", d)
rules = {n: met(f"runs/eval/e357_{n}_sec0/val.json") for n in ("lo", "lo_near", "lo_all")}
hi = met("runs/eval/e357_hi_sec0/val.json")
best = max(rules, key=lambda n: rules[n]["topo_F1"])
b = rules[best]["topo_F1"]; h = hi["topo_F1"]
print(f"best rule (topo F1): {best} {b:.4f}; hi {h:.4f}; gap {h-b:.4f}; half-gap line {b + 0.5*(h-b):.4f}")
res = {}
for S in (0, 1):
    p = f"runs/e357_p_s{S}/tok_card_thr0.5_sec0.json"
    if not os.path.exists(p):
        print("seed", S, "missing"); continue
    m = met(p)
    g1 = (m["topo_F1"] - b) / (h - b)
    jl = m.get("link_junction_F1", float("nan"))
    res[S] = (g1, jl)
    print(f"seed {S}: topo F1 {m['topo_F1']:.4f} (gap share {g1:.3f}), junction F1 {m['junction_F1']:.4f}, junction-link P {m.get('link_junction_P', float('nan')):.4f} R {m.get('link_junction_R', float('nan')):.4f} F1 {jl:.4f}, short-link P {m.get('link_short_P', float('nan')):.4f} R {m.get('link_short_R', float('nan')):.4f}, lanes_found {m['lanes_found']:.4f}, F1@1.0 {m['F1@1.0']:.4f}, dup_frac {m['dup_frac']:.4f}, lane_long_gt {m.get('lane_long_gt', m.get('link_lane_long_n_gt'))}")
if len(res) == 2:
    fail = all(r[0] < 0.5 for r in res.values()) or all(r[1] < 0.50 for r in res.values())
    ok = all(r[0] >= 0.5 and r[1] >= 0.75 for r in res.values())
    print("FAIL" if fail else ("PASS" if ok else "INCONCLUSIVE"))
