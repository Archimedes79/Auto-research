import json
f = lambda r: sum(json.load(open(f"runs/{r}/jmaps_val.json"))["keypoints"][f"kp_ours_tol3.0_noang_{k}"]["f1"] for k in ("ent", "ext")) / 2
for S in (0, 1):
    print(S, "E307", round(f(f"e307_n_s{S}"), 4), "E312 maps after pair stage", round(f(f"e312_p_s{S}_maps"), 4), "E316 maps after pair stage", round(f(f"e316_p_s{S}_maps"), 4))
