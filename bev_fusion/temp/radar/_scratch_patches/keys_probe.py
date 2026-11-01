import json
m = json.load(open("runs/t10_dual_s0/eval/metrics.json"))["metrics"]
ks = [k for k in m if k.startswith("det/dt+0.00/vehicle/")]
print(len(m), len(ks))
print("third slot values:", sorted({k.split("/")[3] for k in ks}))
print([k for k in ks if k.split("/")[3] != "all"][:8])
print(sorted({"/".join(k.split("/")[:2]) for k in m})[:12])
print([k for k in m if "occl" in k or "vis" in k or "range" in k][:8])
