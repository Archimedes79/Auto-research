import json, os
R = os.environ["BFLC_ROOT"]
keys = ["topo_F1", "F1@1.0", "dup_frac", "vert100_dp", "junction_F1", "lanes_found", "Q"]
for s in (0, 1):
    for tag, f in (("snap", "ends_card_snap_thr0.1.json"), ("unsnap", "ends_card_thr0.1.json")):
        out = {}
        for run in (f"e312_p_s{s}", f"e341_p_s{s}"):
            p = os.path.join(R, "runs", run, f)
            if not os.path.exists(p):
                out[run] = None
                continue
            h = json.load(open(p))["numbers"]["head"]
            out[run] = h
        a, b = out[f"e312_p_s{s}"], out[f"e341_p_s{s}"]
        if a and b:
            print(s, tag, " ".join(f"{k} {b[k]:.4f}({b[k]-a[k]:+.4f})" for k in keys if k in a and k in b))
        else:
            print(s, tag, "missing", [k for k, v in out.items() if v is None])
