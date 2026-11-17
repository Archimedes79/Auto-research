import os
p = os.path.join(os.environ["BFLC_ROOT"], "training/grid/test_lane_map.py")
s = open(p, encoding="utf-8").read()
old = '("the cap is 512 again (the real maps are cut)", "MAX_SAMPLES = 704 ", "MAX_SAMPLES = 512 ", "check"),'
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, '("the cap is 200 (the real maps are cut)", "MAX_SAMPLES = 704 ", "MAX_SAMPLES = 200 ", "check"),'))
print("ok")
