"""E362: the verdict table by the written rule of the card (section 0), both seeds."""
import json, os
def met(p):
    d = json.load(open(p)); return d.get("metrics", d)
best, hi = 0.7011, 0.9666
K = ("topo_F1", "topo_P", "topo_R", "junction_F1", "lanes_found", "F1@1.0", "dup_frac", "vert100_dp", "link_short_P", "link_short_R", "link_short_F1", "link_junction_P", "link_junction_R", "link_junction_F1")
res = {}
for S in (0, 1):
    r = f"runs/e362_p4_s{S}"
    for dec in ("rec", "cond03"):
        for ro in ("sec0", "gtsec"):
            p = f"{r}/tok2_card_{dec}_{ro}.json"
            if not os.path.exists(p): print(S, dec, ro, "missing"); continue
            m = met(p)
            if dec == "rec" and ro == "sec0":
                g1 = (m["topo_F1"] - best) / (hi - best); res[S] = (g1, m["link_junction_F1"])
            print(f"seed {S} {dec:6s} {ro:5s}", " ".join(f"{k}={m[k]:.4f}" for k in K if k in m), (f"gap_share={g1:.3f}" if dec == "rec" and ro == "sec0" else ""))
    p = f"{r}/tok2_card_rec_drop0.17_s0_sec0.json"
    if os.path.exists(p):
        m = met(p); print(f"seed {S} rec degraded(17%) sec0", " ".join(f"{k}={m[k]:.4f}" for k in ("topo_F1", "lanes_found", "link_junction_F1") if k in m))
if len(res) == 2:
    fail = all(v[0] < 0.5 for v in res.values()) or all(v[1] < 0.50 for v in res.values())
    ok = all(v[0] >= 0.5 and v[1] >= 0.75 for v in res.values())
    print("VERDICT:", "FAIL" if fail else ("PASS" if ok else "INCONCLUSIVE"), {S: (round(v[0], 3), round(v[1], 4)) for S, v in res.items()})
