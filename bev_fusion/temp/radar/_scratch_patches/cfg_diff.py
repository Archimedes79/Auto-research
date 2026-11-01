import json
a = json.load(open("training/configs/t10_dual_s0.json")); b = json.load(open("training/configs/t11_vel_hurdle_acc2.json"))
for k in sorted(set(a) | set(b)):
    if k in ("train_scenes", "val_scenes"): continue
    if a.get(k) != b.get(k): print(k, ":", json.dumps(a.get(k))[:150], "->", json.dumps(b.get(k))[:200])
