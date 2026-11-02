"""scratch of the t16 report mail (D46): the plain-text body of the mail from the stored evaluation tables + the narrative file temp/t16/mail_narrative.txt (written by hand at the end).
    python3 bev_fusion/temp/t16/build_mail.py > bev_fusion/results/t16/mail.txt"""
import os
import re
import subprocess

RUNS = [("t16_full_s0", "baseline"), ("t16_ctl_s0", "control"), ("t17_data", "more data"), ("t18_fix", "bundle")]
ROWS = [("drivable area IoU", "seg_fov/seg/dt+0.00/drivable_area/all/all/iou@0"), ("walkway IoU", "seg_fov/seg/dt+0.00/walkway/all/all/iou@0"), ("car park IoU", "seg_fov/seg/dt+0.00/carpark_area/all/all/iou@0"),
        ("ped. crossing IoU", "seg_fov/seg/dt+0.00/ped_crossing/all/all/iou@0"), ("lane divider IoU", "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0.5"), ("road divider IoU", "seg_fov/seg/dt+0.00/road_divider/all/all/iou@1"),
        ("mAP (fusion area)", "detf/dt+0.00/mean/all/mAP"), ("vehicle AP@2 m", "detf/dt+0.00/vehicle/all/ap@2"), ("pedestrian AP@2 m", "detf/dt+0.00/pedestrian/all/ap@2"),
        ("vel. error moving centres", "vel/dt+0.00/centre/field/moving/epe"), ("vel. error static cells", "vel/dt+0.00/cells/static/epe")]


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
out = []
narr = open("bev_fusion/temp/t16/mail_narrative.txt", encoding="utf-8").read() if os.path.exists("bev_fusion/temp/t16/mail_narrative.txt") else "(narrative missing)\n"
out.append(narr.rstrip("\n"))
out.append("")
out.append("RESULTS (validation set, 1000 frames of 82 scenes, ground truth v8, fusion-area scoring, ONE seed per run; higher is better except the two velocity rows)")
w = max(len(n) for n, _ in ROWS) + 2
out.append(" " * w + "".join(f"{l:>12s}" for _, l, _ in have))
for name, key in ROWS:
    out.append(f"{name:{w}s}" + "".join(f"{t[key]:12.3f}" if key in t else f"{'-':>12s}" for _, _, t in have))
head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
out.append("")
out.append(f"Repository: github.com/Archimedes79/Auto-research, main at {head} when this mail was written.")
print("\n".join(out))
