"""scratch (t22 gate A (ii)): the gate rows of the card from the metrics.json of the stored reference, the live-code twin and the stand-in.

    python temp/t22/chain/compare_gate.py REFERENCE_RUN [TWIN_RUN] STANDIN_RUN [--all]

Prints, per row: reference, twin, stand-in, the stand-in minus the twin (the student effect) and the stand-in minus the reference, and the pass flag of the card (not
worse than the reference by the margin: 0.05 for lane divider IoU@1, drivable and mAP of the fusion area, 0.08 for vehicle AP@2 at 40 - 60 m). --all lists every
row where the stand-in differs from the twin (or from the reference when there is no twin) by more than 0.02. The numbers are read from RUN/eval/metrics.json, not typed.
"""
import json
import os
import sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
show_all = "--all" in sys.argv
if len(args) not in (2, 3):
    sys.exit(__doc__)
ref_run, stand_run = args[0], args[-1]
twin_run = args[1] if len(args) == 3 else None


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, p + k + "/"))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[p + k] = v
    return out


def load(run):
    return flat(json.load(open(os.path.join(run, "eval", "metrics.json")))["metrics"])


ref, stand = load(ref_run), load(stand_run)
twin = load(twin_run) if twin_run else None

GATE = [
    ("lane divider IoU@1", "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@1", 0.05),
    ("drivable IoU@0", "seg_fov/seg/dt+0.00/drivable_area/all/all/iou@0", 0.05),
    ("drivable IoU@0.5", "seg_fov/seg/dt+0.00/drivable_area/all/all/iou@0.5", 0.05),
    ("mAP fusion area (detf)", "detf/dt+0.00/mean/all/mAP", 0.05),
    ("vehicle AP@2 40-60 m (detf)", "detf/dt+0.00/vehicle/40-60/ap@2", 0.08),
    ("vehicle AP@2 40-60 m (det)", "det/dt+0.00/vehicle/40-60/ap@2", 0.08),
]
OTHER = [
    ("walkway IoU@0", "seg_fov/seg/dt+0.00/walkway/all/all/iou@0"),
    ("ped crossing IoU@0", "seg_fov/seg/dt+0.00/ped_crossing/all/all/iou@0"),
    ("lane divider IoU@0.5", "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0.5"),
    ("road divider IoU@1", "seg_fov/seg/dt+0.00/road_divider/all/all/iou@1"),
    ("centreline IoU@1", "seg_fov/seg/dt+0.00/centreline/all/all/iou@1"),
    ("vehicle seg IoU@0 dt0", "seg_obj/seg/dt+0.00/vehicle/all/all/iou@0"),
    ("vehicle seg IoU@0 dt+0.5", "seg_obj/seg/dt+0.50/vehicle/all/all/iou@0"),
    ("mAP det (all)", "det/dt+0.00/mean/all/mAP"),
    ("vehicle AP@2 all (det)", "det/dt+0.00/vehicle/all/ap@2"),
]


def g(d, k):
    return d.get(("metrics/" + k) if ("metrics/" + k) in d else k)


def row(name, key, margin=None):
    r, s = ref.get(key), stand.get(key)
    t = twin.get(key) if twin else None
    if r is None or s is None:
        print(f"{name:34s} missing: ref {r} stand-in {s}")
        return
    cells = [f"{name:34s}", f"ref {r:7.4f}"]
    cells.append(f"twin {t:7.4f}" if t is not None else "twin    -   ")
    cells.append(f"stand-in {s:7.4f}")
    cells.append(f"s-t {s - t:+7.4f}" if t is not None else "s-t    -   ")
    cells.append(f"s-ref {s - r:+7.4f}")
    if margin is not None:
        cells.append("PASS" if s - r >= -margin else f"FAIL (margin {margin})")
    print("  ".join(cells))


print(f"reference {ref_run}   twin {twin_run}   stand-in {stand_run}   (VAL 1000 frames, one seed each; point estimates)")
print("-- gate rows of the card")
for name, key, m in GATE:
    row(name, key, m)
print("-- other rows (no margin)")
for name, key in OTHER:
    row(name, key)

if show_all:
    base = twin if twin else ref
    def is_score(k):
        last = k.split("/")[-1]
        return not (last.startswith("n_") or last in ("n", "ms", "ms_per_frame") or "timing" in k or "memory" in k)

    diffs = [(abs(stand[k] - base[k]), k) for k in stand if k in base and is_score(k) and abs(stand[k] - base[k]) > 0.02 and abs(base[k]) < 50]
    diffs.sort(reverse=True)
    print(f"-- {len(diffs)} rows with |stand-in - {'twin' if twin else 'reference'}| > 0.02 (largest 40)")
    for d, k in diffs[:40]:
        print(f"{k:90s} base {base[k]:9.4f} stand-in {stand[k]:9.4f} diff {stand[k] - base[k]:+9.4f}")
