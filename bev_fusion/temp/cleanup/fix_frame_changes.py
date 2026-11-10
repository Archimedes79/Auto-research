import sys
p = sys.argv[1]
lines = open(p, encoding="utf-8", newline="").read().split("\n")
out, mode = [], 0
NEW = ("- 2026-11-10 (clean-base round, BEV system lead, data path and tests): **no behaviour change** (dead code and comments only; D73 lifted the freeze, so no number of any run is touched): "
       "`training/freeze.py` loses `select` (no caller) and `fingerprint` (only its own test used it; the `hashlib` import with it; `tests/test_freeze_probe.py` now checks \"the model it was given is "
       "untouched\" with `snapshot` / `unchanged`), `training/nuscenes_dataset.py` loses the unused `Quaternion` import and gets two corrected path texts, `training/feature_cache.py` loses a dead "
       "reference in its docstring; `training/temporal.py` and `temp/radar/radar_recon_test.py` moved to `studies/early/` (only studies import them); the recipes of the closed experiments moved from "
       "`temp/` to `studies/recipes/`.")
for l in lines:
    if l.startswith("<<<<<<< "):
        mode = 1
        continue
    if l.startswith("=======") and mode == 1:
        mode = 2
        out.append(NEW)
        continue
    if l.startswith(">>>>>>> ") and mode == 2:
        mode = 0
        continue
    if mode == 2:
        continue
    out.append(l)
open(p, "w", encoding="utf-8", newline="").write("\n".join(out))
print("ok", sum(1 for l in out if l.startswith("- 2026-11-10 (clean-base")), "lines of today")
