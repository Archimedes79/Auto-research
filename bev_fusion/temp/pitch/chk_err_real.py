import json, sys
sys.path.insert(0, ".")
import evaluation.bev_errors as E
m = json.load(open("runs/t22_B_s0/eval/metrics.json"))
ks = [k for k in m if k.startswith("det/dt+0.00/vehicle/all/") and ("n_" in k.rsplit("/", 1)[1] or "recall" in k)]
print(ks[:12])
rows, tot, nf = E.error_rows("runs/t22_B_s0", 1000, "", ("vehicle",), 2.0, 0.0)
print("min-score 0:", tot)
for k in ks[:12]:
    print(k, m[k])
