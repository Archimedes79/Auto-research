import glob, json, os
ROOT = "/mnt/c/Auto-research/bev_fusion"
cf = {}
for p in sorted(glob.glob(f"{ROOT}/training/configs/*.json") + glob.glob(f"{ROOT}/runs/*/config.json")):
    try: c = json.load(open(p, encoding="utf-8"))
    except Exception: continue
    n = os.path.basename(os.path.dirname(p)) if "/runs/" in p else os.path.splitext(os.path.basename(p))[0]
    cf.setdefault(n, c)
live = [n for n, c in cf.items() if isinstance(c, dict) and not c.get("feature_cache") and not (c.get("model") or {}).get("camera_off")]
print(len(cf), "distinct; no feature_cache and camera on:", len(live), live[:20])
nocache_off = [n for n, c in cf.items() if isinstance(c, dict) and not c.get("feature_cache") and (c.get("model") or {}).get("camera_off")]
print("no cache, camera_off:", nocache_off)
