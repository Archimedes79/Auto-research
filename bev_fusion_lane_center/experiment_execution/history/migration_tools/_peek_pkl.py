import pickle
import sys

R = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
for p in ("e94_a/graphs_val_clean_or.pkl", "e94_a/graphs_val_clean.pkl", "e120_a/refined_val.pkl", "_scene_ids.pkl"):
    try:
        z = pickle.load(open(f"{R}/{p}", "rb"))
    except Exception as e:
        print(p, "ERR", repr(e)[:100])
        continue
    print(p, type(z).__name__, (len(z) if hasattr(z, "__len__") else ""), (list(z.keys())[:8] if isinstance(z, dict) else ""))
    if isinstance(z, dict):
        for k, v in list(z.items())[:6]:
            print("   ", k, type(v).__name__, (len(v) if hasattr(v, "__len__") else v))
        it = z.get("items") or z.get("graphs")
        if it is not None:
            e = it[0]
            print("   first:", type(e).__name__, list(e.keys())[:12] if isinstance(e, dict) else "")
    elif isinstance(z, list):
        e = z[0]
        print("   first:", type(e).__name__, list(e.keys())[:12] if isinstance(e, dict) else "")
