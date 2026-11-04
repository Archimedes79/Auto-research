import json
m = json.load(open("runs/t10_dual_s0_gen/eval/metrics.json"))
d = m["metrics"]
ck = sorted(k for k in d if k.startswith("curve/"))
print(len(ck), "curve keys; step", d.get("info/step"), "n", d.get("info/n_samples"))
for k in ck:
    if k.startswith("curve/dt+0.00/") and k.rsplit("/", 1)[-1] in ("ap", "roc_auc", "ece", "epe", "r2") and ("lane_divider" in k or "drivable" in k or "vehicle" in k or "velocity" in k):
        print(k, round(d[k], 4))
c = json.load(open("runs/t10_dual_s0_gen/eval/card.json"))
print("card has curve keys:", sum(k.startswith("curve/") for k in c["metrics"]), "| better entries for curve:", sum(k.startswith("curve/") for k in c["better"]))
