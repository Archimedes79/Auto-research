import os
root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
p = os.path.join(root, "plan", "head_design.md")
s = open(p, encoding="utf-8").read()
a = s.index("\n## 7. Losses: which make sense, and the radar branch (2026-11-03)")
new = open(os.path.join(root, "temp", "head_session_2026-11-01", "sec7b.txt"), encoding="utf-8").read()
open(p, "w", encoding="utf-8", newline="\n").write(s[:a].rstrip("\n") + "\n\n" + new)
print("section 7 replaced; sections:", sum(1 for l in open(p, encoding="utf-8") if l.startswith("## ")))
