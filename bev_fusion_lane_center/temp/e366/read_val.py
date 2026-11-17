import json, glob
ks = ("topo_F1", "topo_P", "topo_R", "F1@1.0", "lanes_found", "junction_F1", "dup_frac", "vert100_dp", "Q")
for S in (0, 1):
    for f in sorted(glob.glob(f"runs/e365_s1_s{S}/token_card_rec_lt0.1_kt0.6_nng*.json")):
        d = json.load(open(f)); m = d["metrics"]; g = m.get("graph", m); info = d.get("info", {})
        print(S, f.split("/")[-1], {k: round(g[k], 4) for k in ks if k in g})
        for key in ("paired_vs_record", "paired_vs_plain_b", "paired_vs_bsec0_close"):
            pv = info.get(key)
            if pv and "topo_F1" in pv:
                print("    ", key, {k: round(v, 4) for k, v in pv["topo_F1"].items() if k in ("a", "b", "diff", "lo", "hi")})
        for key in ("record_numbers", "record_vs_stored", "decode_variant"):
            if key in info: print("    ", key, json.dumps(info[key])[:300])
