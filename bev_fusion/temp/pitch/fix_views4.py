p = "evaluation/bev_views.py"
s = open(p, encoding="utf-8").read()
def rep(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)
rep("c = radar_points(rad).xy if False else radar_points(rad).color", "c = radar_points(rad).color")
rep('''    sh2 = next(x for x in past_box_shapes([{"group": "vehicle", "cx": 10.0, "cy": 0.0, "length": 4.0, "width": 2.0, "yaw": 0.0}], {"T_e2g": T_then}, {"T_e2g": T_now2}) if True else [])
''', '''    sh2 = next(x for x in past_box_shapes([{"group": "vehicle", "cx": 10.0, "cy": 0.0, "length": 4.0, "width": 2.0, "yaw": 0.0}], {"T_e2g": T_then}, {"T_e2g": T_now2}) if x.kind == "polygon")
''')
a = s.index('\n\nif __name__ == "__main__":\n    import sys')
s = s[:a] + "\n"
open(p, "w", encoding="utf-8").write(s)
p = "evaluation/bev_selftest.py"
s = open(p, encoding="utf-8").read()
anchor = '@check("taps of the new modules'
assert s.count(anchor) == 1
s = s.replace(anchor, '''@check("bev_views helpers: radar opacity and Doppler direction, speed-direction colours, the moving layer, faded past boxes through the poses, the error colours of boxes")
def _():
    from evaluation import bev_views
    bev_views.selftest()


''' + anchor)
open(p, "w", encoding="utf-8").write(s)
