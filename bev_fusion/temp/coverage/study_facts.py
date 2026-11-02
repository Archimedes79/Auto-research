"""The derived numbers of the footprint-share record, computed from temp/coverage/footprint_share.csv (so the text does not hold typed numbers).
Run from bev_fusion (WSL, gcar python): python temp/coverage/study_facts.py -> prints json and writes temp/coverage/study_facts.json"""
import csv
import json

import numpy as np

rows = [r for r in csv.DictReader(open("temp/coverage/footprint_share.csv")) if int(r["val"]) == 1 and float(r["in_grid"]) > 0]
f = np.array([float(r["f_fus"]) for r in rows])
fno = np.array([float(r["f_fusnoocc"]) for r in rows])
g = np.array([r["group"] for r in rows])
rng = np.array([float(r["rng"]) for r in rows])
cf = np.array([int(r["c_fus"]) for r in rows])
ob = np.array([int(r["observed"]) for r in rows])
n = len(f)
pc = lambda k, d=n: round(100.0 * k / d, 1)
o = {"objects": n, "frames": len({(r["scene"], r["sample"]) for r in rows})}
o["f_zero"] = [int((f <= 0).sum()), pc((f <= 0).sum())]
o["f_one"] = [int((f >= 1).sum()), pc((f >= 1).sum())]
o["f_between"] = [int(((f > 0) & (f < 1)).sum()), pc(((f > 0) & (f < 1)).sum())]
o["states_0.1_0.5"] = {"dont_care": [int((f < 0.1).sum()), pc((f < 0.1).sum())], "partial": [int(((f >= 0.1) & (f < 0.5)).sum()), pc(((f >= 0.1) & (f < 0.5)).sum())], "care": [int((f >= 0.5).sum()), pc((f >= 0.5).sum())]}
o["grazing_0_to_0.1"] = int(((f > 0) & (f < 0.1)).sum())
o["centre_vs_f"] = {"centre_in": int(cf.sum()), "centre_in_f_below_0.5": int(((cf == 1) & (f < 0.5)).sum()), "centre_out": int((cf == 0).sum()), "centre_out_f_at_least_0.5": int(((cf == 0) & (f >= 0.5)).sum())}
o["centre_vs_f"]["disagree_total"] = o["centre_vs_f"]["centre_in_f_below_0.5"] + o["centre_vs_f"]["centre_out_f_at_least_0.5"]
o["centre_vs_f"]["disagree_percent"] = pc(o["centre_vs_f"]["disagree_total"])
o["occlusion_loses"] = {"at_least_0.1": [int(((fno - f) >= 0.1).sum()), pc(((fno - f) >= 0.1).sum())], "at_least_0.5": [int(((fno - f) >= 0.5).sum()), pc(((fno - f) >= 0.5).sum())]}
o["by_group"] = {k: {"n": int((g == k).sum()), "f_zero": pc(((g == k) & (f <= 0)).sum(), (g == k).sum()), "care_0.5": pc(((g == k) & (f >= 0.5)).sum(), (g == k).sum())} for k in sorted(set(g))}
o["observed_flag"] = {"observed_f_below_0.5": int(((ob == 1) & (f < 0.5)).sum()), "unobserved_f_at_least_0.5": int(((ob == 0) & (f >= 0.5)).sum()), "unobserved": int((ob == 0).sum())}
o["care_by_range_percent"] = {f"{lo}-{hi}": [int(((rng >= lo) & (rng < hi)).sum()), pc(((rng >= lo) & (rng < hi) & (f >= 0.5)).sum(), ((rng >= lo) & (rng < hi)).sum())] for lo, hi in ((0, 20), (20, 40), (40, 60))}
step = {}
for a, b in ((0.4, 0.5), (0.5, 0.6), (0.6, 0.75), (0.75, 0.9)):
    step[f"{a}->{b}"] = int(((f >= a) & (f < b)).sum())
o["objects_between_thresholds"] = step
json.dump(o, open("temp/coverage/study_facts.json", "w"), indent=1)
print(json.dumps(o, indent=1))
