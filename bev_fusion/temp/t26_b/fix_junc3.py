p = "evaluation/bev_junction.py"
s = open(p, encoding="utf-8").read()
a = s.index("    # GT paint inside the junction that the network does not draw")
b = s.index("    # the dilation does not carry the inside out:")
s = s[:a] + s[b:]
old = '''    # through the real add_sample:'''
new = '''    # and the same the other way round: a GT cell inside must not confirm a predicted cell outside
    p3, g3, f3 = junction_free({"lane_divider": gt2}, {"lane_divider": pred2}, fov, inter)       # prediction just outside (ix 9), GT just inside (ix 10)
    mj, mo = SegMeter(grid), SegMeter(grid)
    mj.add(p3, g3, mask=f3)
    mo.add({"lane_divider": gt2}, {"lane_divider": pred2}, mask=fov)
    rj, ro = mj.result()[0.0]["lane_divider"]["all"]["all"], mo.result()[0.0]["lane_divider"]["all"]["all"]
    assert ro["iou@0.5"] == 1.0 and rj["iou@0.5"] == 0.0 and rj["n_gt"] == 0, (ro, rj)
    # through the real add_sample:'''
assert old in s
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)
