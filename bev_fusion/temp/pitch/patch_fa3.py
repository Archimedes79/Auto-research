p = "evaluation/bev_fusion_area.py"
s = open(p, encoding="utf-8").read()
def rep(old, new):
    global s
    assert old in s, old[:70]
    s = s.replace(old, new, 1)
rep('coverage_classes(footprint_share(b, area), footprint_share(b, cam), None if pts is None else pts[1])',
    'coverage_classes(corner_share(rb["corners"], area), corner_share(rb["corners"], cam), None if pts is None else pts[1])')
# selftest updates
rep('''    keep, scored = prepare(dets, gts, area, include_partial=False)''', '''    shares = [corner_share(corners(b), area) for b in gts]
    assert [gt_class(f) for f in shares] == ["care", "partial", "dont_care"], shares
    keep, scored = prepare(dets, gts, shares, area, include_partial=False)''')
rep('''    keep2, scored2 = prepare(dets, gts, area, include_partial=True)''', '''    keep2, scored2 = prepare(dets, gts, shares, area, include_partial=True)''')
rep('''    keep3, _ = prepare([dict(box(6.5, 0), score=1.0)], [care, box(8.0, 0.0, l=1.0)], area, include_partial=False)
    assert len(keep3) == 1''', '''    keep3, _ = prepare([dict(box(6.5, 0), score=1.0)], [care, box(8.0, 0.0, l=1.0)], [1.0, 1.0], area, include_partial=False)
    assert len(keep3) == 1
    edge = box(59.5, 0)                                                        # a neutral box just outside the window (x >= 60) still neutralises a detection next to it
    area2 = np.ones((140, 104), bool)
    area2[118:] = False
    k4, s4 = prepare([dict(box(58.5, 0), score=1.0)], [dict(box(60.5, 0))], [0.0], area2, include_partial=False)
    assert k4 == [] and s4 == [], "a detection within 2 m of a neutral box outside the window is ignored"
    assert footprint_share(dict(box(5, 0), length=float("nan")), area) == 0.0, "a NaN box is outside"
    assert 0.0 <= footprint_share(dict(box(5, 0), length=1e6, width=1e6), area) <= 1.0, "a huge predicted box is clamped, not sampled"''')
rep('''    care, partial, dc = box(5, 0), box(11.4, 5), box(30, -5)''', '''    def corners(b):                                                           # the four consecutive corners of a box dict, as the GT stores them
        c, s_ = math.cos(b["yaw"]), math.sin(b["yaw"])
        return [(b["cx"] + sx * b["length"] / 2 * c - sy * b["width"] / 2 * s_, b["cy"] + sx * b["length"] / 2 * s_ + sy * b["width"] / 2 * c) for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
    care, partial, dc = box(5, 0), box(11.4, 5), box(30, -5)''')
rep('''    assert gt_class(0.5) == "care"''', '''    assert abs(corner_share([(9, -1), (9, 1), (11, 1), (11, -1)], area) - 0.5) < 0.03 and corner_share([(1, -1), (1, 1), (3, 1), (3, -1)], area) == 1.0, "corner share"
    assert gt_class(0.5) == "care"''')
open(p, "w", encoding="utf-8").write(s)
