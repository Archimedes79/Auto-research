"""The card metrics of finished arms side by side (t21 / D51; any run with eval/card.json): the velocity error at moving centres, the front mAP and the
lane / drivable / walkway IoUs of each run, cold (1000 frames, `card.json`) and warm (whole scenes, state carried, `card_warm.json`).
Reads stored files only; numbers are never typed by hand. A metric a run does not have is printed as nan.

    python evaluation/arm_guards.py [run name ...]            (from the bev_fusion folder; names without runs/; default: the t21 arms)
Test: tests/test_stream_tools.py."""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
KEYS = [
    ("EPE moving centres (lower)", "vel/dt+0.00/centre/field/moving/epe"),
    ("mAP fusion area", "detf/dt+0.00/mean/all/mAP"),
    ("lane divider IoU@1 (fov)", "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@1"),
    ("lane divider IoU@0.5 (fov)", "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0.5"),
    ("drivable IoU@1 (fov)", "seg_fov/seg/dt+0.00/drivable_area/all/all/iou@1"),
    ("walkway IoU@1 (fov)", "seg_fov/seg/dt+0.00/walkway/all/all/iou@1"),
]
DEFAULT = ["t20_base_s0", "t21_ctl_s0", "t21_rec_s0b", "t21_pair_s0", "t21_pair4_s0"]


def load(name, tag="", root=ROOT):
    p = os.path.join(root, "runs", name, "eval", f"card{tag}.json")
    return json.load(open(p))["metrics"] if os.path.exists(p) else None


def table(names, root=ROOT):
    """the printed table as a list of lines (cold, then warm; a block is left out when no run has its card)"""
    out = []
    for tag in ("", "_warm"):
        cols = [(n, load(n, tag, root)) for n in names]
        cols = [(n, m) for n, m in cols if m]
        if not cols:
            continue
        out.append(f"\n=== card{tag}.json (VAL, {'1000 frames, cold per sample' if not tag else 'whole scenes, state carried'}) ===")
        out.append(f"{'':32}" + "".join(f"{n:>14}" for n, _ in cols))
        for label, k in KEYS:
            out.append(f"{label:32}" + "".join(f"{m.get(k, float('nan')):14.4f}" for _, m in cols))
    return out


if __name__ == "__main__":
    print("\n".join(table(sys.argv[1:] or DEFAULT)))
