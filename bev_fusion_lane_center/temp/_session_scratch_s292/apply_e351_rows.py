import io
import re

rows = io.open("temp/e351/readme_rows.md", encoding="utf-8").read().split("\n")
new = [r for r in rows if r.startswith("| `jtiming.py`") or r.startswith("| `jgraph.py`")]
assert len(new) == 2, len(new)

p = "evaluation/README.md"
L = io.open(p, encoding="utf-8", newline="").read().split("\n")
idx = [i for i, l in enumerate(L) if l.startswith("| `jtiming.py`")]
assert len(idx) == 1, idx
assert not any(l.startswith("| `jgraph.py`") for l in L)
L[idx[0]:idx[0] + 1] = new
io.open(p, "w", encoding="utf-8", newline="").write("\n".join(L))

p = "tools/maintenance/run_gates.sh"
s = io.open(p, encoding="utf-8", newline="").read()
anchor = 'run "timing: junction system ms per frame, section markers on a toy JEnds (E342 part B, CPU)"  jtiming.py selftest\n'
assert s.count(anchor) == 1
line = [l for l in io.open("temp/e351/run_gates_line.txt", encoding="utf-8").read().split("\n") if l.startswith("run ")]
assert len(line) == 1
s = s.replace(anchor, anchor + line[0] + "\n")
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
