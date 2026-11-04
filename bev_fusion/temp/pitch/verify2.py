import csv, json
# (A) 3-class mAP and two_wheeler AP@2, ctl vs dual
for run in ("t8_ctl_s0", "t10_dual_s0", "t8_bevfeat_s0"):
    m = json.load(open(f"runs/{run}/eval/metrics.json"))["metrics"]
    print(run, "mAP3", round(m.get("det/dt+0.00/mean/all/mAP", float("nan")), 4),
          "two_wheeler ap@2", round(m.get("det/dt+0.00/two_wheeler/all/ap@2", float("nan")), 4),
          "veh ap@2", round(m.get("det/dt+0.00/vehicle/all/ap@2", float("nan")), 4),
          "ped ap@2", round(m.get("det/dt+0.00/pedestrian/all/ap@2", float("nan")), 4))
# (B) executed lane runs with a prediction and a rule
rows = list(csv.DictReader(open("../bev_fusion_lane_center/experiment_execution/LEDGER.csv", newline="", encoding="utf-8")))
ex = [r for r in rows if r["status"] != "planned"]
print("lane executed", len(ex), "with prediction", sum(1 for r in ex if r["prediction"].strip()), "with rule", sum(1 for r in ex if r["rule"].strip()),
      "with both", sum(1 for r in ex if r["prediction"].strip() and r["rule"].strip()))
rows = list(csv.DictReader(open("records/LEDGER.csv", newline="", encoding="utf-8")))
print("bev rows", [(r["id"], r["status"], bool(r["prediction"].strip()), bool(r["rule"].strip())) for r in rows])
# (E) t10.dual_s1 record
d = json.load(open("records/runs/t10.dual_s1/run.json"))
print({k: d.get(k) for k in ("status", "attempts", "started", "commit", "code_hash", "exit", "duration_s")})
