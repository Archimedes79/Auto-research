"""E363 stop rule (iii), the FIXTURE: the pair run of E324 s with B's OWN node head (`runs/e363_b_s0`, 0 tensors changed by the surgery) dumped by `jends_eval.py dump` must reproduce the stored dump `runs/e324_s_s0/ends_val.pkl`: every array of every frame allclose (atol 2e-3, bool / int arrays equal).
Writes temp/e363/fixture_result.json (`pass`, the largest absolute difference per field, the frame count).   usage: wsl bash tools/env/_py14.sh fixture.py"""
import json
import os
import pickle
import numpy as np

ROOT = os.environ["BFLC_ROOT"]
ATOL = 2e-3
a = pickle.load(open(f"{ROOT}/runs/e324_s_s0/ends_val.pkl", "rb"))
b = pickle.load(open(f"{ROOT}/runs/e363_b_s0/ends_val.pkl", "rb"))
res = dict(frames_stored=len(a), frames_new=len(b), atol=ATOL, max_abs={}, bad=[])
ok = len(a) == len(b) == 873
for x, y in zip(a, b):
    if x["row"] != y["row"] or x["token"] != y["token"]:
        ok = False
        res["bad"].append(f"row {x['row']} vs {y['row']}")
        continue
    for k in x:
        if k in ("row", "token"):
            continue
        u, v = x[k], y.get(k)
        if u is None or v is None:
            if (u is None) != (v is None):
                ok = False
                res["bad"].append(f"{k} None mismatch row {x['row']}")
            continue
        u, v = np.asarray(u), np.asarray(v)
        if u.shape != v.shape:
            ok = False
            res["bad"].append(f"{k} shape row {x['row']}")
            continue
        if u.dtype == bool or np.issubdtype(u.dtype, np.integer):
            same = np.array_equal(u, v)
            d = 0.0 if same else 1.0
        else:
            d = float(np.nanmax(np.abs(u.astype(float) - v.astype(float)))) if u.size else 0.0
            same = d <= ATOL
        res["max_abs"][k] = max(res["max_abs"].get(k, 0.0), d)
        if not same:
            ok = False
            res["bad"].append(f"{k} row {x['row']} diff {d:.4g}")
res["bad"] = res["bad"][:20]
res["pass"] = bool(ok)
json.dump(res, open(f"{ROOT}/temp/e363/fixture_result.json", "w"), indent=1)
print("FIXTURE", "PASS" if ok else "FAIL", {k: round(v, 6) for k, v in res["max_abs"].items()}, res["bad"][:5])
