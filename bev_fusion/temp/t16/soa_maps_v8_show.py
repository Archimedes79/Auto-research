"""throw-away: print the stored-count gate of soa_maps_v8 output files. usage: soa_maps_v8_show.py a.json [b.json ...]"""
import json
import sys

for p in sys.argv[1:]:
    d = json.load(open(p))
    g = d["stored_check"]
    m = d["meta"]
    print(p.split("/")[-1], f"{m['seconds_per_frame']:.1f} s/frame (model {m['seconds_model']:.1f})", "dts:", m.get("dts_decoded"))
    if not g:
        print("  no stored check")
        continue
    print("  frames", g["n_frames"], "max count diff", g["max_count_diff"], "max score diff", g["max_score_diff"], "max err diff", g["max_err_diff"])
    for c in g["frames"]:
        print("  ", c["label"], c.get("closest_drivable_cell"), c.get("cells_within_1e-4_of_threshold"), "keys", c["keys_compared"], "differing", c["keys_differing"], c["differing"][:4], "err", round(c["max_err_diff"], 4), "score", c["max_score_diff"])
