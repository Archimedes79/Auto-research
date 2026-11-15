"""Print the card metrics of t21 arms next to the reference (t20.base_s0) and the control (t21.ctl_s0, when its eval exists).
    ~/miniconda3/envs/gcar/bin/python temp/t21/guards.py [run names ...]      (from bev_fusion/; names without runs/ and without a prefix)"""
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
KEYS = [
    ("EPE moving centres (lower)", "vel/dt+0.00/centre/field/moving/epe"),
    ("mAP fusion area", "detf/dt+0.00/mean/all/mAP"),
    ("lane divider IoU@1 (fov)", "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@1"),
    ("lane divider IoU@0.5 (fov)", "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0.5"),
    ("drivable IoU@1 (fov)", "seg_fov/seg/dt+0.00/drivable_area/all/all/iou@1"),
    ("walkway IoU@1 (fov)", "seg_fov/seg/dt+0.00/walkway/all/all/iou@1"),
]


def load(name, tag=""):
    p = os.path.join(ROOT, "runs", name, "eval", f"card{tag}.json")
    return json.load(open(p))["metrics"] if os.path.exists(p) else None


def main():
    names = sys.argv[1:] or ["t20_base_s0", "t21_ctl_s0", "t21_rec_s0b", "t21_pair_s0", "t21_pair4_s0"]
    for tag in ("", "_warm"):
        cols = [(n, load(n, tag)) for n in names]
        cols = [(n, m) for n, m in cols if m]
        if not cols:
            continue
        print(f"\n=== card{tag}.json (VAL, {'1000 frames, cold per sample' if not tag else 'whole scenes, state carried'}) ===")
        print(f"{'':32}" + "".join(f"{n:>14}" for n, _ in cols))
        for label, k in KEYS:
            print(f"{label:32}" + "".join(f"{m.get(k, float('nan')):14.4f}" for _, m in cols))


main()
