"""scratch: the base's velocity-complete card on GT v13 against the stored card on GT v9: copy to the stable path, list the rows that moved"""
import json
import math
import os
import shutil

B = "/mnt/c/Auto-research/bev_fusion/"
new_p = B + "temp/t26/base_run/eval/by_n/card_n1000.json"
os.makedirs(B + "runs/t26_base/eval", exist_ok=True)
shutil.copy(new_p, B + "runs/t26_base/eval/card.json")
for extra in ("metrics_n1000.json",):
    shutil.copy(B + "temp/t26/base_run/eval/by_n/" + extra, B + "runs/t26_base/eval/metrics.json")
new = json.load(open(new_p))
old = json.load(open(B + "runs/t20_base_s0/eval/card.json"))
mn, mo = new.get("metrics", new), old.get("metrics", old)
print("gt new:", (new.get("protocol") or {}).get("gt_signature"), "| gt old:", (old.get("protocol") or {}).get("gt_signature"), "| frames equal:", new.get("frames") == old.get("frames"), "| n keys new/old:", len(mn), len(mo))
common = [k for k in mn if k in mo and isinstance(mn[k], (int, float)) and isinstance(mo[k], (int, float)) and math.isfinite(mn[k]) and math.isfinite(mo[k])]
diffs = sorted(((abs(mn[k] - mo[k]), k, mo[k], mn[k]) for k in common), reverse=True)
print("common numeric keys:", len(common), "| changed (>1e-9):", sum(1 for d in diffs if d[0] > 1e-9))
print("largest changes:")
for d, k, a, b in diffs[:25]:
    print(f"  {d:9.5f}  {k}  {a:.5f} -> {b:.5f}")
for k in ("vel/dt+0.00/centre/field/moving/epe", "vel/dt+0.00/centre/field/static/epe", "detf/dt+0.00/mean/all/mAP", "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0.5"):
    print(k, "old", mo.get(k), "new", mn.get(k))
print("only in old:", len([k for k in mo if k not in mn]), "| only in new:", len([k for k in mn if k not in mo]))
