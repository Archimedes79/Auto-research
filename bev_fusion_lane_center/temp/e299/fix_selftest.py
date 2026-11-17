p = "evaluation/jtime.py"
s = open(p, encoding="utf-8").read()
a = s.index("    # the ego moves 2 m forward"); b = s.index("    # round trip:")
s = s[:a] + "    fa, fb = lane_frame(info(0.0, (0.0, 0.0))), lane_frame(info(0.7, (2.0, -1.0)))\n    Wp = warp(np.array([[0.0, 10.0, 0.0, 1.0]]), fa, fb)\n    assert abs(np.linalg.norm(Wp[0, 2:]) - 1.0) < 1e-9 and np.linalg.norm(Wp[0, :2] - np.array([0.0, 10.0])) > 0.5, Wp     # a rotated and moved ego: the heading stays a unit vector, the position changes\n" + s[b:]
open(p, "w", encoding="utf-8").write(s); print("ok")
