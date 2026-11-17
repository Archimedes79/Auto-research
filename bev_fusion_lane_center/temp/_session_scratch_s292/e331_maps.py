import json, os
R = os.environ["BFLC_ROOT"] + "/runs/"
for s in (0, 1):
    a = json.load(open(f"{R}e331_e_s{s}_maps/jmaps_val.json"))["kp_f1_best_threshold"]["ours"]
    b = json.load(open(f"{R}e312_p_s{s}_maps/jmaps_val.json"))["kp_f1_best_threshold"]["ours"] if os.path.exists(f"{R}e312_p_s{s}_maps/jmaps_val.json") else None
    c = json.load(open(f"{R}e307_n_s{s}/jmaps_val.json"))["kp_f1_best_threshold"]["ours"]
    print(s, "e331 maps kp F1", round(a, 4), "e312 maps", None if b is None else round(b, 4), "e307", round(c, 4))
