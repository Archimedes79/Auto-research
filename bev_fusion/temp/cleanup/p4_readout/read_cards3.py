import json, os, re
B = os.path.dirname(os.path.abspath(__file__))
F = {"a": "a/eval/card.json", "b": "b/eval/card_ctl_zero_radar.json", "c": "c/eval/card.json"}
C = {k: json.load(open(os.path.join(B, v)))["metrics"] for k, v in F.items()}
keys = [k for k in C["a"] if k.startswith(("detf/dt+0.00/vehicle/all/", "detf/dt+0.00/pedestrian/all/", "vel", "obj", "motion"))]
print([k for k in C["a"] if "epe" in k.lower()][:6])
for k in keys:
    print(f"{k:70s}", *[f"{C[v].get(k):9.4f}" if isinstance(C[v].get(k), (int, float)) else f"{str(C[v].get(k)):>9s}" for v in "abc"])
