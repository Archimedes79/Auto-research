import json, os
R = os.environ["BFLC_ROOT"]
keys = ["topo_F1", "F1@1.0", "dup_frac", "vert100_dp", "junction_F1", "lanes_found"]
for s in (0, 1):
    for tag, f in (("snap", "ends_card_snap_thr0.1.json"), ("unsnap", "ends_card_thr0.1.json")):
        out = {}
        for run in (f"e312_p_s{s}", f"e334_o_s{s}"):
            p = os.path.join(R, "runs", run, f)
            out[run] = json.load(open(p))["numbers"]["head"] if os.path.exists(p) else None
        a, b = out[f"e312_p_s{s}"], out[f"e334_o_s{s}"]
        print(s, tag, "E334", " ".join(f"{k} {b[k]:.4f}({b[k]-a[k]:+.4f})" for k in keys), "| E312 topo", f"{a['topo_F1']:.4f}")
