import json, glob
ks = ("topo_F1", "F1@1.0", "lanes_found", "junction_F1", "junction_P", "junction_R", "dup_frac", "vert100_dp", "topo_P", "topo_R")
for r in ("e373_cb_s0", "e373_cb_s1", "e367_b25_s0", "e367_b25_s1"):
    f = glob.glob(f"runs/{r}/token_card_rec_lt0.3_ltl0.2_kt0.6_nng.json")[0]
    d = json.load(open(f)); g = d["metrics"].get("graph", d["metrics"]); info = d["info"]
    rc = info["ref_cell"]; rn = rc.get("numbers", rc)
    rg = rn.get("graph", rn) if isinstance(rn, dict) else {}
    print(r, "cell", {k: round(g[k], 4) for k in ks if k in g})
    print("   ref ", {k: round(rg[k], 4) for k in ks if k in rg})
    pv = info["paired_vs_ref_cell"]
    for k in ("topo_F1", "junction_F1", "F1@1.0", "lanes_found"):
        if k in pv: print("   paired", k, {x: round(pv[k][x], 4) for x in ("diff", "lo", "hi") if x in pv[k]})
