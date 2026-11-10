"""One-off (t2_block_swaps): table of all finished arms vs the reference `t2_ref_s0` from the metric files, with the paired scene
bootstrap (evaluate_run.compare) and the verdict of the card's decision rule. Nothing typed by hand.
Usage (bev_fusion folder): python temp/study/collect.py > experiments/t2_block_swaps/results/arms.md"""
import contextlib
import io
import json
import os
import sys

sys.path.insert(0, ".")
from evaluation import bev_legacy as ev  # noqa: E402

ARMS = ["ref_s0", "ref_s1", "t3_radar_only", "t3_cam_zsum", "t3_cam_adapter", "t3_adapter", "t3_zsum", "t3_ref_long", "radar_off", "trunk_rand", "lift_p", "camtemp_none", "fusion_gate", "radar_pillars",
        "trunk_second", "trunk_small", "lift_s", "camtemp_gru", "fusion_cmx"]
KEYS = [("mAP2", "mAP2", 3), ("veh AP@1", "det/dt+0.00/vehicle/all/ap@1", 3), ("ped AP@1", "det/dt+0.00/pedestrian/all/ap@1", 3),
        ("veh AP@2", "det/dt+0.00/vehicle/all/ap@2", 3), ("ped AP@2", "det/dt+0.00/pedestrian/all/ap@2", 3),
        ("mAP3", "det/dt+0.00/mean/all/mAP", 3), ("veh AOE", "det/dt+0.00/vehicle/all/aoe", 2),
        ("veh AVE", "det/dt+0.00/vehicle/all/ave", 2),
        ("drivable", "seg_fov/seg/dt+0.00/drivable_area/all/all/iou@0", 3),
        ("walkway", "seg_fov/seg/dt+0.00/walkway/all/all/iou@0", 3),
        ("lane div@0.5", "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0.5", 3),
        ("val loss", "val_loss/total", 2)]
BOOT = {"mAP2": None, "veh AP@1": "vehicle AP@1", "ped AP@1": "pedestrian AP@1", "veh AP@2": "vehicle AP@2", "ped AP@2": "pedestrian AP@2", "drivable": "drivable_area iou@0",
        "walkway": "walkway iou@0", "lane div@0.5": "lane_divider iou@0.5"}


def load(arm):
    p = f"runs/{arm if arm.startswith('t3_') else 't2_' + arm}/eval/metrics.json"
    if not os.path.exists(p):
        return None
    m = json.load(open(p))["metrics"]
    # mAP2: vehicle + pedestrian only (the third class, two_wheeler, has 174 GT in 1000 frames and swings +-0.15 AP between seeds)
    m["mAP2"] = sum(m[f"det/dt+0.00/{c}/all/ap@{t}"] for c in ("vehicle", "pedestrian") for t in (0.5, 1, 2, 4)) / 8
    return m


M = {a: load(a) for a in ARMS}
M = {a: m for a, m in M.items() if m}
ref = M.get("ref_s0")
if ref is None:
    sys.exit("no ref_s0 yet")
band = 0.010
if "ref_s1" in M:
    band = max(0.010, 2 * max(abs(M["ref_s1"][k] - ref[k]) for _, k, _ in [KEYS[i] for i in (0, 1, 2, 8, 9, 10)]))
n = int(ref["info/n_samples"])
print(f"# t2_block_swaps: arms vs `ref_s0` (cold VAL, {n} keyframes of {int(ref['info/n_scenes'])} scenes, EMA, 3000 steps)\n")
print(f"Noise band for the verdict: {band:.3f}" + (" (2 x seed spread, `ref_s1`)" if "ref_s1" in M else " (no second seed yet: 0.010)")
      + ". Difference = arm - `ref_s0`; interval = paired scene bootstrap (1000 draws).\n")
print("| arm | " + " | ".join(k for k, _, _ in KEYS) + " |")
print("|---|" + "---|" * len(KEYS))
for a, m in M.items():
    print(f"| {a} | " + " | ".join(f"{m.get(k, float('nan')):.{d}f}" for _, k, d in KEYS) + " |")
print("\n## Differences and verdicts\n")
print("| arm | " + " | ".join(k for k, _, _ in KEYS[:11]) + " | verdict |")
print("|---|" + "---|" * 12)
for a, m in M.items():
    if a == "ref_s0":
        continue
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res, _ = ev.compare("runs/t2_ref_s0", f"runs/{a if a.startswith('t3_') else 't2_' + a}", n_boot=300)
    cells, worse, better = [], [], []
    for lab, key, d in KEYS[:11]:
        diff = m.get(key, float("nan")) - ref.get(key, float("nan"))
        mark = ""
        bk = BOOT.get(lab)
        if bk and bk in res:
            _, lo, hi, _, _ = res[bk]
            sig = lo > 0 or hi < 0
            mark = "*" if sig else ""
            if sig and abs(diff) > band and lab not in ("veh AOE", "veh AVE"):
                (better if diff > 0 else worse).append(lab)
        elif lab == "mAP2" and abs(diff) > band:
            (better if diff > 0 else worse).append(lab + " (no bootstrap)")
        cells.append(f"{diff:+.{d}f}{mark}")
    verdict = "WORSE: " + ", ".join(worse) if worse else ("BETTER: " + ", ".join(better) if better else "TIE")
    if worse and better:
        verdict = "MIXED: worse " + ", ".join(worse) + "; better " + ", ".join(better)
    print(f"| {a} | " + " | ".join(cells) + f" | {verdict} |")
print("\n`*` = paired bootstrap interval excludes 0. Velocity / heading errors (AVE, AOE) are lower-better and have no bootstrap; "
      "they are not used in the verdict.")
