import json, os
R = os.environ["BFLC_ROOT"] + "/runs/"
K = ("topo_F1", "F1@1.0", "junction_F1", "dup_frac", "vert100_dp", "lanes_found", "topo_P", "topo_R")
for ro, f in (("unsnapped", "ends_card_thr0.1.json"), ("snapped", "ends_card_snap_thr0.1.json")):
    for s in (0, 1):
        a = json.load(open(f"{R}e331_e_s{s}/{f}"))["numbers"]["head"]; c = json.load(open(f"{R}e312_p_s{s}/{f}"))["numbers"]["head"]
        j = json.load(open(f"{R}e331_e_s{s}/{f}")); jc = json.load(open(f"{R}e312_p_s{s}/{f}"))
        print(f"{ro:9s} s{s} " + " ".join(f"{k} {a[k]:.4f} ({c[k]:.4f}, {a[k]-c[k]:+.4f})" for k in K[:6]) + f" | curves {j['curves']} ({jc['curves']})")
