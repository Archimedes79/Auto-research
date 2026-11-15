"""scratch: the t23 read-out table (card experiments/t23_lane_line_loss): prediction rows and guards of runs/<new>/eval/card.json against the control, point estimates from the stored cards.
python temp/lane_tests/read_t23.py [new run=t23_C_s0] [control=t21_ctl_s0]
The paired intervals come from `bev_suite.py compare runs/<control> runs/<new> --curves --fusion --n 1000` (run separately). Missing keys are printed as MISSING, never silently skipped."""
import json
import os
import sys

new = sys.argv[1] if len(sys.argv) > 1 else "t23_C_s0"
ctl = sys.argv[2] if len(sys.argv) > 2 else "t21_ctl_s0"


def card(run):
    p = f"runs/{run}/eval/card.json"
    return json.load(open(p))["metrics"] if os.path.exists(p) else None


a, b = card(ctl), card(new)
if a is None or b is None:
    print("missing card:", ctl if a is None else "", new if b is None else "")
    sys.exit(2)
ROWS = [  # (label, key, direction: +1 higher better, -1 lower better, 0 reported, prediction threshold or None)
    ("PRED lane divider IoU@0.5", "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0.5", +1, 0.449),
    ("PRED lane divider IoU@1", "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@1", +1, 0.562),
    ("PRED lane divider AP (ranking)", "curve/dt+0.00/lane_divider/ap", +1, None),
    ("PRED lane divider roc_auc", "curve/dt+0.00/lane_divider/roc_auc", +1, None),
    ("guard road divider IoU@1", "seg_fov/seg/dt+0.00/road_divider/all/all/iou@1", +1, None),
    ("guard drivable IoU@0", "seg_fov/seg/dt+0.00/drivable_area/all/all/iou@0", +1, None),
    ("guard walkway IoU@0", "seg_fov/seg/dt+0.00/walkway/all/all/iou@0", +1, None),
    ("guard ped crossing IoU@0", "seg_fov/seg/dt+0.00/ped_crossing/all/all/iou@0", +1, None),
    ("guard mAP fusion area", "detf/dt+0.00/mean/all/mAP", +1, None),
    ("guard velocity EPE moving (lower)", "vel/dt+0.00/centre/field/moving/epe", -1, None),
    ("guard centreline IoU@0.5 (lane centre)", "seg_fov/seg/dt+0.00/centreline/all/all/iou@0.5", +1, None),
    ("guard centreline IoU@1 (lane centre)", "seg_fov/seg/dt+0.00/centreline/all/all/iou@1", +1, None),
    ("guard centreline AP (lane centre)", "curve/dt+0.00/centreline/ap", +1, None),
    ("report lane divider zc F1@0.5", "seg_fov/seg/dt+0.00/lane_divider:zc/all/all/f1@0.5", 0, None),
    ("report lane divider zc p@0.5", "seg_fov/seg/dt+0.00/lane_divider:zc/all/all/p@0.5", 0, None),
    ("report lane divider zc r@0.5", "seg_fov/seg/dt+0.00/lane_divider:zc/all/all/r@0.5", 0, None),
]
vehicle = sorted(k for k in a if "vehicle" in k and k.endswith("/all/all/iou@0") and "dt+0.00" in k)
print(f"{'row':44s} {'control':>9s} {'new':>9s} {'diff':>8s}  note   (control {ctl}, new {new}, VAL 1000 frames)")
for label, key, d, thr in ROWS + [(f"guard vehicle {k.split('/')[0]} IoU@0", k, +1, None) for k in vehicle[:2]]:
    va, vb = a.get(key), b.get(key)
    if va is None or vb is None:
        print(f"{label:44s} {'MISSING':>9s}  key {key}")
        continue
    note = ""
    if thr is not None:
        note = "meets prediction" if vb >= thr else f"below prediction {thr}"
    print(f"{label:44s} {va:9.4f} {vb:9.4f} {vb - va:+8.4f}  {note}")
