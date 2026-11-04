import json
c = json.load(open("runs/t10_dual_s0_gen/eval/card.json"))
d = c["provenance"]["data"]
u = d.get("use", {})
print({k: v for k, v in d.items() if k != "use"})
print({k: (v if k != "scene_names" else v[:3] + ["..."]) for k, v in u.items()})
