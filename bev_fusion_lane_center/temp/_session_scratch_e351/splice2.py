"""scratch: replace the selftest section (from its '# ---- 5.' header to `def main():`) of evaluation/jgraph.py by new_tests.txt"""
import io
import os

R = os.environ["BFLC_ROOT"]
p = os.path.join(R, "evaluation", "jgraph.py")
s = io.open(p, encoding="utf-8", newline="").read()
new = io.open(os.path.join(R, "temp", "_session_scratch_e351", "new_tests.txt"), encoding="utf-8", newline="").read()
a = s.index("# ---------------------------------------------------------------- 5. the selftest")
b = s.index("def main():")
io.open(p, "w", encoding="utf-8", newline="").write(s[:a] + new + s[b:])
print("spliced", len(s[a:b]), "->", len(new))
