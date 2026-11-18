import json, sys, numpy as np
f = sys.argv[1]
j = json.load(open(f"/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/{f}/result.json"))
perts = list(j["perts"])
keys = [k for k in j["perts"][perts[0]] if not k.startswith("_")]
print("variants over", len(perts), "perturbation types (mean over types); failure phase | clean phase")
for k in keys:
    fl = np.mean([j["perts"][p][k]["fail"]["lane_F1"] for p in perts]); fp = np.mean([j["perts"][p][k]["fail"]["lane_P"] for p in perts]); fr = np.mean([j["perts"][p][k]["fail"]["lane_R"] for p in perts])
    rs = np.mean([j["perts"][p][k]["fail"]["route_success"] for p in perts])
    cl = np.mean([j["perts"][p][k]["clean"]["lane_F1"] for p in perts]); cr = np.mean([j["perts"][p][k]["clean"]["route_success"] for p in perts])
    print(f"{k:20s} fail laneF1 {fl:.3f} (P {fp:.3f} R {fr:.3f}) route {rs:.3f} | clean laneF1 {cl:.3f} route {cr:.3f}")
mon = {p: j["perts"][p]["_monitor"] for p in perts}
print("monitor AUROC(agreement)", {p: round(m["auroc_mass"], 3) for p, m in mon.items()})
# best absolute threshold on pooled agreement
ag = np.concatenate([j["perts"][p]["_mon_data"]["ag"] for p in perts]); fl_ = np.concatenate([j["perts"][p]["_mon_data"]["fail"] for p in perts])
best = None
for t in np.linspace(ag.min(), ag.max(), 200):
    pred = ag < t
    acc = (pred == (fl_ == 1)).mean()
    if best is None or acc > best[0]: best = (acc, t, pred[fl_ == 1].mean(), pred[fl_ == 0].mean())
print("best absolute threshold", best)
