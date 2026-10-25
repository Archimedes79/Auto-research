"""Loop 18 / E232 (one-off): table of the long joint fine-tune (4800 steps, builder loss at full weight): route success (clean half / failure half) and graph quality of the failure series with the spatial hold, base route head s0 and base builder (E214 evaluation) against arm c (control: only the route head learns) and arm a (the route loss reaches the builder), route seed 0.
Usage (gcar): p18_j3_table.py   -> prints and writes runs/e18/e232_table.json"""
import bflc_paths as BP  # project paths from the location of this checkout
import json
import pickle

E17 = BP.ROOT + "/runs/e17"
R18 = BP.ROOT + "/runs/e18"
arms = [("base s0", "e18_j2", "e18_j2_route", "base", "p16_f2hr_route_s0"), ("c0", "e18_j3", "e18_j3_route", "c0", "e18_joint2_c0"), ("a0", "e18_j3", "e18_j3_route", "a0", "e18_joint2_a0")]
tags = ("dark_maxdecay_0.95", "blur_maxdecay_0.95", "cam_front3_off_maxdecay_0.95")
names = {"dark_maxdecay_0.95": "night", "blur_maxdecay_0.95": "blur", "cam_front3_off_maxdecay_0.95": "three cameras off"}
res = {}
for t in tags:
    print(f"== {names[t]}")
    for lab, gdir, rdir, sfx, route in arms:
        j = json.load(open(f"{E17}/{rdir}/route_{t}_{sfx}.json"))
        k = f"sigma0_{route}_"
        rc, rf = j[k + "cl"]["route_success"], j[k + "fail"]["route_success"]
        g = pickle.load(open(f"{E17}/{gdir}/{t}_{sfx}.pkl", "rb"))["cards"]
        res[f"{t}|{lab}"] = dict(route_clean=rc, route_fail=rf, S_clean=g["cl"]["S"], S_fail=g["fail"]["S"], F1_fail=g["fail"]["F1@1.0"], found_fail=g["fail"]["lanes_found"])
        print(f"  {lab:8s} route success clean {rc:.3f} failure {rf:.3f} | graph S clean {g['cl']['S']:.3f} failure {g['fail']['S']:.3f} F1@1 {g['fail']['F1@1.0']:.3f} found {g['fail']['lanes_found']:.3f}")
diff = {}
print("\narm a - arm c (failure-half route success; graph S failure half) and arm c - base")
for t in tags:
    a, c, b = res[f"{t}|a0"], res[f"{t}|c0"], res[f"{t}|base s0"]
    diff[t] = dict(a_minus_c_route=a["route_fail"] - c["route_fail"], a_minus_c_S=a["S_fail"] - c["S_fail"], c_minus_base_route=c["route_fail"] - b["route_fail"], c_minus_base_S=c["S_fail"] - b["S_fail"])
    print(f"  {names[t]}: a - c route {diff[t]['a_minus_c_route']:+.3f} S {diff[t]['a_minus_c_S']:+.3f} | c - base route {diff[t]['c_minus_base_route']:+.3f} S {diff[t]['c_minus_base_S']:+.3f}")
n_ok = sum(1 for t in tags if diff[t]["a_minus_c_route"] >= 0.03)
s_ok = all(diff[t]["a_minus_c_S"] >= -0.005 for t in tags)
print(f"rule: a - c >= +0.03 in {n_ok} of 3 types; graph S not lower by more than 0.005 in all types: {s_ok}; met: {n_ok >= 2 and s_ok}")
json.dump(dict(res=res, diff=diff, types_met=n_ok, s_ok=s_ok), open(f"{R18}/e232_table.json", "w"), indent=1)
