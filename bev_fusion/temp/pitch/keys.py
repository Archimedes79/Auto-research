import json
m = json.load(open("runs/t10_dual_s0/eval/metrics.json"))
print({k: (type(v).__name__, len(v) if hasattr(v, "__len__") else v) for k, v in m.items()})
for k, v in m.items():
    if isinstance(v, dict) and len(v) > 20:
        ks = sorted(v)
        print(k, len(ks))
        print("\n".join(x for x in ks if "dt+0.00" in x and "/all/all/" in x and ("divider" in x or "carpark" in x or "crossing" in x))[:1500])
        print([x for x in ks if x.startswith("det/dt+0.00/mean") or "mAP2" in x][:8])
