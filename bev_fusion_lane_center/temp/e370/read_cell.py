import json
for S in (0, 1):
    f = f"runs/e370_b_s{S}/token_card_rec_lt0.1_kt0.6_nng.json"
    try:
        d = json.load(open(f)); g = d["metrics"].get("graph", d["metrics"])
        print(S, {k: round(g[k], 4) for k in ("topo_F1", "F1@1.0", "lanes_found", "junction_F1", "dup_frac")})
    except FileNotFoundError:
        print(S, "not yet")
