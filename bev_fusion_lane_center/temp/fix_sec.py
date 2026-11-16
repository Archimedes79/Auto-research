p = "training/grid/sections.py"
s = open(p, encoding="utf-8").read()
s = s.replace("    pb += [0, 0], pa += [5, 11], pc += [6, 0]", "    pb += [0, 0]\n    pa += [5, 11]\n    pc += [6, 0]")
open(p, "w", encoding="utf-8").write(s)
