import json
a = json.load(open("training/configs/t10_dual_s0.json"))["model"]; b = json.load(open("training/configs/t11_vel_hurdle_acc2.json"))["model"]
print({k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)})
