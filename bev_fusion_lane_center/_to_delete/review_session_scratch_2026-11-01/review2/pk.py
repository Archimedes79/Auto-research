import pickle, json
r = pickle.load(open("/mnt/c/Auto-research/bev_fusion/runs/t8_ctl_s0/eval/scene_records_n1000.pkl", "rb"))
print(len(r["meta"]["frames"]), r["meta"]["gt_signature"])
m = json.load(open("/mnt/c/Auto-research/bev_fusion/runs/t8_ctl_s0/eval/metrics.json"))
print(m.get("gt_signature"), m.get("train_gt_signature"))
