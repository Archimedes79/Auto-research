P = "/mnt/c/Auto-research/bev_fusion/experiments/t26_one_long_run/experiment.md"
s = open(P).read()
old = "The base is BEHIND on all six rows by the written rule except that drivable area (d1 -0.041 against the margin 0.030, interval wholly below 0) is BEHIND too; the gaps"
assert s.count(old) == 1
s = s.replace(old, "By the written rule the base is BEHIND on all six rows (drivable area too: d1 -0.041 against its margin 0.030, the interval wholly below 0); the gaps")
open(P, "w").write(s); print("ok")
