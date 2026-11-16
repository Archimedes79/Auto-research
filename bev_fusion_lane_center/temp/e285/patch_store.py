p = "training/startable.py"
s = open(p, encoding="utf-8").read()
a = "        sig_dir = lane_gt.signature()\n"
b = "        sig_dir = os.path.join(lane_gt.DATA, lane_gt.signature())\n"
assert a in s
open(p, "w", encoding="utf-8").write(s.replace(a, b))
