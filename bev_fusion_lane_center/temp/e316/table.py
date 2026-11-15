import json
def card(r, name="ends_card_snap_thr0.1.json"):
    return json.load(open(f"runs/{r}/{name}"))
keys = ["topo_F1", "lanes_found", "junction_F1", "junction_R", "dup_frac", "F1@1.0", "vert100_dp", "Q"]
for S in (0, 1):
    for r in (f"e312_p_s{S}", f"e316_p_s{S}"):
        c = card(r)
        n = c["numbers"] if "numbers" in c else c
        print(r, {k: round(n["head"][k], 4) if "head" in n and k in n["head"] else None for k in keys} if isinstance(n, dict) and "head" in n else list(n.keys())[:10])
    c = card(f"e316_p_s{S}")
    print(" difference to plain", {k: round(v, 4) for k, v in c["difference"].items()} if isinstance(c["difference"], dict) else c["difference"])
    h = json.load(open(f"runs/e316_p_s{S}/ends_head.json")); g = json.load(open(f"runs/e312_p_s{S}/ends_head.json"))
    print(" ent/ext recall e316", round(h["ent_recall"], 3), round(h["ext_recall"], 3), "e312", round(g["ent_recall"], 3), round(g["ext_recall"], 3), "pair rec/prec", round(h["pair_recall"], 3), round(h["pair_precision"], 3), "e312", round(g["pair_recall"], 3), round(g["pair_precision"], 3))
    for r in (f"e312_p_s{S}", f"e316_p_s{S}"):
        m = json.load(open(f"runs/{r}/maneuver_val.json"))
        print(" maneuver", r, {k: (round(v.get("realised_share", v.get("share", 0)), 4) if isinstance(v, dict) else v) for k, v in list(m.items())[:6]})
