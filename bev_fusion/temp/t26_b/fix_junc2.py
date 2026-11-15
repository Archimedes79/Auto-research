p = "evaluation/bev_junction.py"
s = open(p, encoding="utf-8").read()
old = '''    # the dilation does not carry the inside out:'''
new = '''    # GT paint inside the junction that the network does not draw is no miss either (the GT cut); n_gt counts the cells outside only
    p3, g3, f3 = junction_free({"lane_divider": gt_in}, {"lane_divider": gt}, fov, inter)
    ma, mb = SegMeter(grid), SegMeter(grid)
    ma.add({"lane_divider": gt_in}, {"lane_divider": gt}, mask=fov)
    mb.add(p3, g3, mask=f3)
    ca, cb = ma.result()[0.0]["lane_divider"]["all"]["all"], mb.result()[0.0]["lane_divider"]["all"]["all"]
    assert ca["iou@0"] == 30 / 40 and cb["iou@0"] == 1.0 and cb["n_gt"] == 30 and ca["n_gt"] == 40, (ca, cb)
    # the dilation does not carry the inside out:'''
assert old in s
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)
