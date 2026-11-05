"""coverage of the 1000 frames by the mmdet3d-style infos pickles (CPU, read only)"""
import os, pickle, sys, resource, json
ref = pickle.load(open("runs/t16_full_s0/eval/scene_records_n1000.pkl", "rb"))
toks = set(ref["meta"]["frames"])
print("ref frames", len(toks), flush=True)
for name in ("nuscenes_infos_val.pkl", "nuscenes_infos_val_crn.pkl", "nuscenes_infos_train.pkl", "nuscenes_infos_train_crn.pkl"):
    p = os.path.expanduser("~/data/nuscenes_full/" + name)
    r = pickle.load(open(p, "rb"))
    infos = r["infos"] if isinstance(r, dict) and "infos" in r else r
    print(name, type(r).__name__, len(infos), type(infos[0]).__name__, sorted(infos[0])[:8] if isinstance(infos[0], dict) else "", flush=True)
    key = "token" if "token" in infos[0] else ("sample_token" if "sample_token" in infos[0] else None)
    if key:
        have = {i[key] for i in infos}
        print("   covers", len(toks & have), "of", len(toks), flush=True)
    del r, infos
print("maxrss MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3)
