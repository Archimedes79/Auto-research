import json, os
R = os.environ["BFLC_ROOT"] + "/runs/"
for s in (0, 1):
    for nm, run in (("E312", f"e312_p_s{s}"), ("E332 a", f"e332_pa_s{s}"), ("E331", f"e331_e_s{s}")):
        p = f"{R}{run}/maneuver_val.json"
        if not os.path.exists(p):
            print(s, nm, "no maneuver_val.json"); continue
        v = json.load(open(p))["variants"]
        print(s, nm.ljust(7), {k: (v[k]["realised"], round(v[k]["recall"], 4), round(v[k]["junction_precision"], 3)) for k in ("plain", "p0.1", "p0.1s") if k in v})
