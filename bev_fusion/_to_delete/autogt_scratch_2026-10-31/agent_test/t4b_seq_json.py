"""Workaround outside the guide: tokens of scene-0561 from the raw nuScenes json tables (no devkit), then autogt.load."""
import json, os, time, numpy as np, autogt

t0 = time.time()
root = os.path.expanduser("~/data/nuscenes_full/v1.0-trainval")
scenes = json.load(open(os.path.join(root, "scene.json")))
sc = [s for s in scenes if s["name"] == "scene-0561"][0]
samples = {s["token"]: s for s in json.load(open(os.path.join(root, "sample.json")))}
toks, t = [], sc["first_sample_token"]
while t:
    toks.append(t)
    t = samples[t]["next"]
print("json walk s", round(time.time() - t0, 1), "n keyframes", len(toks), flush=True)
shares, pooled_m, pooled_c, missing = [], 0, 0, 0
for tok in toks:
    try:
        g = autogt.load(tok)
    except Exception as e:
        missing += 1
        if missing == 1:
            print("first load error:", type(e).__name__, str(e)[:200])
        continue
    moving = (np.hypot(*g["velocity"]) > 1.0) & (g["flow_weight"] > 0)
    covered = g["vehicle"] > 0
    n = int(covered.sum())
    if n == 0:
        continue
    m = int((moving & covered).sum())
    shares.append(m / n); pooled_m += m; pooled_c += n
print("loaded", len(toks) - missing, "missing", missing, "frames with vehicle cells", len(shares))
if shares:
    print("mean over frames of share moving among vehicle-covered cells", float(np.mean(shares)))
    print("pooled share", pooled_m / pooled_c, "cells", pooled_c)
print("total s", round(time.time() - t0, 1))
