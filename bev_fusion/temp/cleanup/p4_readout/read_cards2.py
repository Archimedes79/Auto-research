import json, os, re
B = os.path.dirname(os.path.abspath(__file__))
F = {"a": "a/eval/card.json", "b": "b/eval/card_ctl_zero_radar.json", "c": "c/eval/card.json"}
C = {k: json.load(open(os.path.join(B, v)))["metrics"] for k, v in F.items()}
keys = [k for k in C["a"] if re.match(r"(detf|det)/dt\+0\.00/(vehicle|pedestrian|mean)/all/(AP@1|AP@2|mAP|ave_skill_moving)", k) or "epe" in k or k.startswith("vel")]
print(len(C["a"]), "metrics; shown", len(keys))
for k in keys:
    print(f"{k:70s}", *[f"{C[v].get(k):9.4f}" if isinstance(C[v].get(k), (int, float)) else f"{str(C[v].get(k)):>9s}" for v in "abc"])
