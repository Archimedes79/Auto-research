import json, pickle
sp = json.load(open("data/splits.json")) if __import__("os").path.exists("data/splits.json") else {}
print({k: (len(v) if hasattr(v, "__len__") else v) for k, v in sp.items()})
r = pickle.load(open("runs/t10_dual_s0/eval/scene_records_n1000.pkl", "rb"))
print(r.keys() if isinstance(r, dict) else type(r))
det = r["det"]
sc = sorted(det)
print(len(sc), sc[:40])
print({s: len(det[s][0.0]) for s in sc[:40]})
