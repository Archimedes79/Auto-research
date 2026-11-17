p = "temp/e338/verdict_e338.py"
s = open(p, encoding="utf-8").read()
a = "        a = ADD[s][k]\n"
assert s.count(a) == 1
s = s.replace(a, "        if k not in ADD[s]:\n            print(f\"  {k:12s} sc {sc[k]:8.4f}  s {s_[k]:8.4f}  D {d[k]:+.4f}  | (no pre-registered number)\")\n            continue\n        a = ADD[s][k]\n")
open(p, "w", encoding="utf-8", newline="\n").write(s)
