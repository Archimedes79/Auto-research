"""scratch: write section 7 (baseline result) of the t16 card from the stored evaluation tables. Re-run when the control / arms have finished (extend RUNS). Run from the repo root."""
import re

p = "bev_fusion/experiments/t16_full_network/experiment.md"
RUNS = [("t16_full_s0", "t16.full_s0 (baseline)"), ("t16_ctl_s0", "t16.ctl_s0 (control)"), ("t17_data", "t17.data"), ("t18_fix", "t18.fix")]


def table(run, n=1000):
    d = {}
    try:
        for line in open(f"bev_fusion/runs/{run}/eval/by_n/table_n{n}.txt", encoding="utf-8"):
            m = re.match(r"^(\S+)\s+(-?[0-9.]+(?:e-?[0-9]+)?|nan)\s*$", line.rstrip())
            if m:
                d[m.group(1)] = float(m.group(2))
    except OSError:
        pass
    return d


have = [(r, l, table(r)) for r, l in RUNS if table(r)]
spec = [("drivable IoU@0", "seg_fov/seg/dt+0.00/drivable_area/all/all/iou@0"), ("walkway IoU@0", "seg_fov/seg/dt+0.00/walkway/all/all/iou@0"),
        ("carpark IoU@0", "seg_fov/seg/dt+0.00/carpark_area/all/all/iou@0"), ("ped crossing IoU@0", "seg_fov/seg/dt+0.00/ped_crossing/all/all/iou@0"),
        ("lane divider IoU@0.5", "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0.5"), ("road divider IoU@1", "seg_fov/seg/dt+0.00/road_divider/all/all/iou@1"),
        ("mAP, fusion area (detf)", "detf/dt+0.00/mean/all/mAP"), ("vehicle AP@2", "detf/dt+0.00/vehicle/all/ap@2"), ("pedestrian AP@2", "detf/dt+0.00/pedestrian/all/ap@2"),
        ("velocity EPE at moving box centres (m/s)", "vel/dt+0.00/centre/field/moving/epe"), ("zero predictor at the same centres", "vel/dt+0.00/centre/zero/moving/epe"),
        ("EPE vehicle_moving cells", "vel/dt+0.00/cells/vehicle_moving/epe"), ("EPE static cells", "vel/dt+0.00/cells/static/epe")]
lines = ["| metric (VAL, 1000 frames, GT v8, one seed) | " + " | ".join(l for _, l, _ in have) + " |", "|---|" + "---|" * len(have)]
for name, key in spec:
    lines.append(f"| {name} | " + " | ".join(("%.3f" % t[key]) if key in t else "-" for _, _, t in have) + " |")
tab = "\n".join(lines)

text = f"""## 7 Results (written 2026-11-02, updated as runs finish; every number from `runs/<run>/eval/by_n/table_n1000.txt`, never typed by hand)

**Large baseline `t16.full_s0`** (config `t16_full.json`, 6000 steps, 428 train / 82 val scenes, GT v8, started 04:47, finished 11:53, 3.61 s per step, no trainsuite alert; code commit `40e7ff5b3c`, tag `bf/t16.full_s0`).
Validation total 11.25 / 8.27 / 7.89 at steps 2000 / 4000 / 6000; the radar-kept monitor is back at 0.90 (it fell to 0.04 in the 400-step run); the train-minus-val gap of the six map IoUs is 0.091
(drivable 0.040, walkway 0.036, carpark 0.171, crossing 0.072, lane divider 0.125, road divider 0.100; `bev_suite.py generalisation --n 1000`).

{tab}

The motion block's error on static cells settled from 1.26 m/s (400-step run) to the value above: the 400-step number was early-training noise (finding 2). The prediction of section 3 (baseline against control within +-0.02 on the maps,
moving EPE below 4.0 m/s, static EPE at most 0.03 above the control) is judged by hand when the control has finished ("by hand: one seed per arm", no KEEP / CUT).

*Queued behind it (all through the launcher, pinned code):* the extraction of the BEVFusion features for 272 more train scenes, `t17.data` (700 train scenes), `t18.fix` (t17 recipe + the Training chat's hygiene flags + the motion block's own learning rate), the matched control `t16.ctl_s0`, each with an evaluation job.
"""
s = open(p, encoding="utf-8").read()
i = s.index("## 7 ")
s = s[:i] + text
open(p, "w", encoding="utf-8").write(s)
print("section 7 written;", len(have), "runs")
