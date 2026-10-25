"""Loop 18: collect the TEST results of the loop-18 chain (E230.*) into runs/e18/test18_table.json: graph card (final-eval card), route success of the route heads along the edges (exact map and 1 / 2 / 3 m error), per finalist.
Usage (any python): python3 p18_test_table.py"""
import json
import os

R = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
LAD = f"{R}/e123_ladder/cards"
out = {}
fin = [("f1p", "e123_D7_r3_f1", "champion f1 288 / 144 + fork pruning 0.3"), ("j1p", "e123_D7_r3_j1", "single-frame j1 192 / 96 + fork pruning 0.3"), ("rcp", "e123_D7_r3_rc1", "recall variant rc1 + fork pruning 0.4")]
heads = ["p16_f2hr_route_s0", "p16_f2hr_route_s1", "e18_joint_c0", "e18_joint_c1"]
for fid, name, lab in fin:
    cp = f"{LAD}/{name}.final18_{fid}_test_clean.json"
    if not os.path.exists(cp):
        continue
    c = json.load(open(cp, encoding="utf-8-sig"))["metrics"]
    Q = (0.30 * c["F1@1.0"] + 0.20 * c["F1@0.5"] + 0.20 * c["J"] + 0.15 * (1 - c["false_len_frac"])) / 0.85
    row = dict(label=lab, S=c["S"], Q=Q, J=c["J"], F1=c["F1@1.0"], found=c["lanes_found"], v100=c["vert100"], R=c["R@1.0"], P=c["P@1.0"], route={})
    for h in heads:
        p = f"{R}/{h}/route2_test18_{fid}.json"
        if os.path.exists(p):
            r = json.load(open(p))["res"]
            row["route"][h] = {k.split("/")[0]: r[k]["route_success"] for k in r if k.endswith("tau0.5") and not k.startswith("_")}
    out[fid] = row
ctrl = {}
for h in ("e104br", "e104br_s1"):
    p = f"{R}/{h}/route2_test18_ctrl_{h}.json"
    if os.path.exists(p):
        r = json.load(open(p))["res"]
        ctrl[h] = {k.split("/")[0]: r[k]["route_success"] for k in r if k.endswith("tau0.5") and not k.startswith("_")}
out["ctrl"] = ctrl
json.dump(out, open(f"{R}/e18/test18_table.json", "w"), indent=1)
print(json.dumps(out, indent=1)[:3000])
