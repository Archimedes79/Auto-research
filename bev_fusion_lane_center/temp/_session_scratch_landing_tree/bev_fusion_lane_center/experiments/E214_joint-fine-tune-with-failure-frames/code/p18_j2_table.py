"""Loop 18 / E214 (one-off): table of the joint fine-tune arms: route success (clean half / failure half) and graph quality of the failure series with the spatial hold, base route heads and builder (no fine-tune) against arm a (route loss reaches the builder)
and arm c (control: only the route head learns), route-head seeds 0 / 1. Usage (gcar): p18_j2_table.py"""
import bflc_paths as BP  # project paths from the location of this checkout
import json
import pickle

import numpy as np

E17 = BP.ROOT + "/runs/e17"
arms = [("base s0", "base", "p16_f2hr_route_s0"), ("base s1", "base", "p16_f2hr_route_s1"), ("c0", "c0", "e18_joint_c0"), ("a0", "a0", "e18_joint_a0"), ("c1", "c1", "e18_joint_c1"), ("a1", "a1", "e18_joint_a1")]
tags = ("dark_maxdecay_0.95", "blur_maxdecay_0.95", "cam_front3_off_maxdecay_0.95")
res = {}
for t in tags:
    print(f"== {t}")
    for lab, sfx, route in arms:
        j = json.load(open(f"{E17}/e18_j2_route/route_{t}_{sfx}.json"))
        k = f"sigma0_{route}_"
        rc, rf = j[k + "cl"]["route_success"], j[k + "fail"]["route_success"]
        g = pickle.load(open(f"{E17}/e18_j2/{t}_{sfx}.pkl", "rb"))["cards"]
        res[(t, lab)] = (rc, rf, g["cl"]["S"], g["fail"]["S"], g["fail"]["F1@1.0"], g["fail"]["lanes_found"])
        print(f"  {lab:8s} route success clean {rc:.3f} failure {rf:.3f} | graph S clean {g['cl']['S']:.3f} failure {g['fail']['S']:.3f} F1@1 {g['fail']['F1@1.0']:.3f} found {g['fail']['lanes_found']:.2f}")
print("\narm a - arm c (failure-half route success; graph S failure half)")
for t in tags:
    print(f"  {t}: seed 0 route {res[(t, 'a0')][1] - res[(t, 'c0')][1]:+.3f} S {res[(t, 'a0')][3] - res[(t, 'c0')][3]:+.3f} | seed 1 route {res[(t, 'a1')][1] - res[(t, 'c1')][1]:+.3f} S {res[(t, 'a1')][3] - res[(t, 'c1')][3]:+.3f}")
print("arm c - base (continued route-head training with failure frames)")
for t in tags:
    print(f"  {t}: seed 0 {res[(t, 'c0')][1] - res[(t, 'base s0')][1]:+.3f} | seed 1 {res[(t, 'c1')][1] - res[(t, 'base s1')][1]:+.3f}")
