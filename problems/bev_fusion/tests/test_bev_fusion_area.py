"""Tests of evaluation/bev_fusion_area.py (its former self-test).

    python tests/test_bev_fusion_area.py      (from the problem folder, with its module path)
"""
from __future__ import annotations
import math

import numpy as np

from evaluation.bev_fusion_area import camera_area, corner_share, coverage_classes, footprint_share, fusion_area, gt_class, prepare


def selftest():
    cam, rad, occ = (np.zeros((4, 4), np.float32) for _ in range(3))
    cam[:2], rad[1:3], occ[1, 1] = 1, 1, 1                                    # camera rows 0-1, radar rows 1-2, hidden cell (1, 1)
    fa, ca = fusion_area({"camera": cam, "radar": rad, "occluded": occ}), camera_area({"camera": cam, "radar": rad, "occluded": occ})
    assert fa.sum() == 3 and fa[1, 0] and fa[1, 2] and fa[1, 3] and not fa[1, 1] and not fa[0].any() and not fa[2].any(), "fusion area = camera x radar x not hidden"
    assert ca.sum() == 7 and ca[0].all() and not ca[1, 1], "camera area = camera x not hidden"
    area = np.zeros((120, 104), bool)
    area[:20, :] = True                                                       # observed: x < 10 m
    box = lambda cx, cy, l=4.0, w=2.0, yaw=0.0, grp="vehicle": dict(group=grp, cx=cx, cy=cy, length=l, width=w, yaw=yaw)   # noqa: E731
    assert footprint_share(box(5, 0), area) == 1.0 and footprint_share(box(15, 0), area) == 0.0
    assert abs(footprint_share(box(10, 0), area) - 0.5) < 0.04, footprint_share(box(10, 0), area)
    assert footprint_share(box(5, 0, yaw=0.7), area) == 1.0, "a turned box inside"
    assert footprint_share(box(5, 40), area) == 0.0, "outside the grid counts as outside"
    ped = dict(group="pedestrian", cx=10.0, cy=0.0, radius=0.4)
    assert 0.35 < footprint_share(ped, area) < 0.65
    assert abs(corner_share([(9, -1), (9, 1), (11, 1), (11, -1)], area) - 0.5) < 0.03 and corner_share([(1, -1), (1, 1), (3, 1), (3, -1)], area) == 1.0, "corner share"
    assert gt_class(0.5) == "care" and gt_class(0.49) == "partial" and gt_class(0.1) == "partial" and gt_class(0.09) == "dont_care"
    # three GT boxes (care, partial, don't care), a detection on each, plus a false detection inside and one outside the area
    def corners(b):                                                           # the four consecutive corners of a box dict, as the GT stores them
        c, s_ = math.cos(b["yaw"]), math.sin(b["yaw"])
        return [(b["cx"] + sx * b["length"] / 2 * c - sy * b["width"] / 2 * s_, b["cy"] + sx * b["length"] / 2 * s_ + sy * b["width"] / 2 * c) for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
    care, partial, dc = box(5, 0), box(11.4, 5), box(30, -5)
    gts = [care, partial, dc]
    fs = [footprint_share(b, area) for b in gts]
    assert [gt_class(f) for f in fs] == ["care", "partial", "dont_care"], fs
    dets = [dict(b, score=1.0) for b in gts] + [dict(box(6, 20), score=1.0), dict(box(40, 0), score=1.0)]
    shares = [corner_share(corners(b), area) for b in gts]
    assert [gt_class(f) for f in shares] == ["care", "partial", "dont_care"], shares
    keep, scored = prepare(dets, gts, shares, area, include_partial=False)
    # the care box + its detection and the false one inside; the neutral ones and the outside one go
    assert len(scored) == 1 and len(keep) == 2, (len(scored), len(keep))
    keep2, scored2 = prepare(dets, gts, shares, area, include_partial=True)
    assert len(scored2) == 2 and len(keep2) == 3, (len(scored2), len(keep2))
    # a detection on the care box that also sits within 2 m of a neutral box of its class stays (it is near a scored one)
    keep3, _ = prepare([dict(box(6.5, 0), score=1.0)], [care, box(8.0, 0.0, l=1.0)], [1.0, 1.0], area, include_partial=False)
    assert len(keep3) == 1
    area2 = np.ones((140, 104), bool)
    area2[118:] = False
    # a neutral box just outside the window (x >= 60) still neutralises a detection next to it
    k4, s4 = prepare([dict(box(58.5, 0), score=1.0)], [dict(box(60.5, 0))], [0.0], area2, include_partial=False)
    assert k4 == [] and s4 == [], "a detection within 2 m of a neutral box outside the window is ignored"
    assert footprint_share(dict(box(5, 0), length=float("nan")), area) == 0.0, "a NaN box is outside"
    assert 0.0 <= footprint_share(dict(box(5, 0), length=1e6, width=1e6), area) <= 1.0, "a huge predicted box is clamped, not sampled"
    assert coverage_classes(1.0, 1.0, 3) == ["all", "care", "camera", "cam_radar"]
    assert coverage_classes(1.0, 1.0, 0) == ["all", "care", "camera", "cam_only"]
    assert coverage_classes(0.0, 0.0, 2) == ["all", "dont_care", "radar_only"]
    assert coverage_classes(0.3, 0.3, 0) == ["all", "partial", "neither"]
    assert coverage_classes(1.0, 1.0, None) == ["all", "care", "camera"]
    print("bev_fusion_area selftest ok: footprint shares, GT classes, neutral detections, coverage classes")


if __name__ == "__main__":
    selftest()
