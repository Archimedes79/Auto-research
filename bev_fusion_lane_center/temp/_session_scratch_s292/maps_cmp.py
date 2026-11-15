import json, os
R = os.environ["BFLC_ROOT"] + "/runs/"
for s in (0, 1):
    row = {}
    for name, run in (("E307 maps (base)", f"e307_n_s{s}"), ("E316 maps", f"e316_p_s{s}_maps"), ("E320 a", f"e320_a_s{s}_maps"), ("E324 s", f"e324_s_s{s}_maps")):
        p = R + run + "/jmaps_val.json"
        if not os.path.exists(p):
            p2 = R + run + "/jmaps_val.json"
            continue
        d = json.load(open(p))
        row[name] = d["kp_f1_best_threshold"]
    print(s, {k: (v if not isinstance(v, dict) else {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()}) for k, v in row.items()})
