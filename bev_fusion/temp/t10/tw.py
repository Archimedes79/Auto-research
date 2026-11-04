import json
for run in ("t8_ctl_s0", "t8_ctl_s1", "t10_dual_s0", "t10_dual_s1", "t8_bevfeat_s0", "t8_bevfeat_s1"):
    m = json.load(open(f"runs/{run}/eval/metrics.json"))["metrics"]
    g = lambda k: round(m.get(k, float("nan")), 4)
    print(run, "mAP3", g("det/dt+0.00/mean/all/mAP"), "veh AP@2", g("det/dt+0.00/vehicle/all/ap@2"), "ped AP@2", g("det/dt+0.00/pedestrian/all/ap@2"),
          "two_wheeler AP@2", g("det/dt+0.00/two_wheeler/all/ap@2"), "veh AVE", g("det/dt+0.00/vehicle/all/ave"))
