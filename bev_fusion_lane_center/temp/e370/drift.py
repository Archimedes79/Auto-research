import json
ks = ("topo_F1", "topo_P", "topo_R", "F1@1.0", "lanes_found", "dup_frac")
for S in (0, 1):
    print("seed", S)
    for tag in ("step0_", "step500_", "step1000_", "step3000_", "step6000_", ""):
        f = f"runs/e370_b_s{S}/token_card_rec_{tag}train_every3.json"
        d = json.load(open(f)); g = d["metrics"].get("graph", d["metrics"])
        print("  ", (tag or "final_")[:-1], {k: round(g[k], 4) for k in ks})
