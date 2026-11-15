import json
m = json.load(open("temp/t22/runs/probe/metrics.json"))
print({k: v for k, v in m.items() if k != "val"})
print({k: round(v, 3) for k, v in m["val"].items()})
