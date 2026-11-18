import json
def g(p):
    c = json.load(open(p, encoding="utf-8"))["p"]["graph"]
    return {k: round(float(c[k]), 4) for k in ("topo_F1", "junction_F1", "F1_1", "lanes_found", "dup_frac", "vert100") if k in c}
for s in (0, 1):
    for arm in ("c2", "a"):
        for rd in ("plain", "close1.5"):
            print(s, arm, rd, g(f"runs/e292/e292_{arm}_s{s}.{rd}_t0.5/card.json"))
    print(s, "B close", g(f"runs/e275/e273_p_s{s}.close1.5_t0.5/card.json"))
    print(s, "B plain", g(f"runs/e275/e273_p_s{s}.plain_t0.5/card.json"))
