import json, os
R = os.environ["BFLC_ROOT"] + "/runs/"
def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict): out.update(flat(v, pre + k + "."))
        elif isinstance(v, (int, float)): out[pre + k] = v
    return out
for s in (0, 1):
    b = flat(json.load(open(f"{R}e275/e273_p_s{s}.close1.5_t0.5/card.json")))
    c = flat(json.load(open(f"{R}e292/e292_c2_s{s}.close1.5_t0.5/card.json")))
    keys = [k for k in b if ("lane_acc" in k or "topo_F1" in k or k.endswith("F1@1.0")) and k in c]
    print(s, {k: (round(b[k], 4), round(c[k], 4)) for k in keys[:12]})
