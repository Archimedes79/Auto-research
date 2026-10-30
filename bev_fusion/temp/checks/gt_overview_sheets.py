"""Overview sheets of the sample GT pictures for a review (2026-10-30): the ten mini scenes of view_gt_samples.py, stacked
3 per sheet with a legend, plus a table (scene, location, time of day, boxes, ignored, stationary, share of observed boxes).
Run from the bev_fusion folder (after view_gt_samples.py):
    wsl ~/miniconda3/envs/gcar/bin/python temp/checks/gt_overview_sheets.py <samples dir> <out dir>
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "training")
from gt_bev import NuScenesGT  # noqa: E402
from lift_pd import Grid  # noqa: E402

PICKS = [("scene-0061", 8), ("scene-0103", 12), ("scene-0553", 10), ("scene-0655", 20), ("scene-0757", 10),
         ("scene-0796", 12), ("scene-0916", 8), ("scene-1077", 8), ("scene-1094", 10), ("scene-1100", 8)]
LEGEND = ["camera: 3D boxes (vehicle red, pedestrian magenta, two-wheeler orange; white rim = stationary, fixed in the world; "
          "grey dashed = no lidar/radar point, ignored); map lines (lane divider yellow, road divider orange, centre-line cyan)",
          "semantic BEV 0.5 m (x forward 0-60 m, y left +-26 m): drivable grey, walkway purple, car park blue, crossing light, "
          "stop line red, dividers, centre-lines with direction arrows",
          "object BEV: boxes with heading, velocity of 1 s (cyan arrows), occluded cells blue, ignored boxes yellow, outside "
          "camera + radar dark, white line = camera field of view"]


def main(samples, out):
    src = NuScenesGT(os.path.expanduser("~/data/nuscenes"), "v1.0-mini")
    G = Grid()
    rows = []
    for name, k in PICKS:
        sc = next(s for s in src.nusc.scene if s["name"] == name)
        tok = sc["first_sample_token"]
        for _ in range(k):
            tok = src.nusc.get("sample", tok)["next"]
        gt = src.keyframe_gt(tok, G)
        bx = gt["meta"]["boxes"]
        loc = src.nusc.get("log", sc["log_token"])["location"]
        night = "night" in sc["description"].lower()
        rows.append((name, k, loc, "night" if night else "day", len(bx), sum(not b["observed"] for b in bx),
                     sum(b["stationary"] for b in bx), float(gt["drivable_area"].mean()), float(gt["camera"].mean())))
    table = ["scene | keyframe | location | light | boxes | ignored | stationary | drivable share | camera-covered share"]
    table += [f"{r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7]:.0%} | {r[8]:.0%}" for r in rows]
    open(os.path.join(out, "gt_overview_table.txt"), "w").write("\n".join(table) + "\n")
    print("\n".join(table))
    imgs = [Image.open(os.path.join(samples, f"gt_sample_{n}_k{k:02d}.png")).convert("RGB") for n, k in PICKS]
    w = imgs[0].width
    groups = [imgs[0:3], imgs[3:6], imgs[6:8], imgs[8:10]]
    for gi, grp in enumerate(groups):
        head = 78
        sheet = Image.new("RGB", (w, head + sum(i.height for i in grp)), "white")
        d = ImageDraw.Draw(sheet)
        d.text((8, 4), f"GT overview, sheet {gi + 1} of {len(groups)} - nuScenes mini, generator training/gt_bev.py", fill="black")
        for li, t in enumerate(LEGEND):
            d.text((8, 22 + 18 * li), t, fill=(60, 60, 60))
        y = head
        for im in grp:
            sheet.paste(im, (0, y))
            y += im.height
        p = os.path.join(out, f"gt_overview_sheet{gi + 1}.png")
        sheet.save(p)
        print("saved", p)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
