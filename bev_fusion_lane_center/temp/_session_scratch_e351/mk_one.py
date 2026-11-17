"""scratch: the dtype mutant g01 again, on the final jgraph.py (the planted cases now also check the KIND of the difference)"""
import os

R = os.environ["BFLC_ROOT"]
src = open(os.path.join(R, "evaluation", "jgraph.py"), encoding="utf-8", newline="").read()
old = "    if a.device != b.device or a.dtype != b.dtype or a.shape != b.shape:"
assert src.count(old) == 1
d = os.path.join(R, "temp", "_session_scratch_e351", "mut_final", "g01b_dtype_ignored")
os.makedirs(d, exist_ok=True)
open(os.path.join(d, "jgraph.py"), "w", encoding="utf-8", newline="").write(src.replace(old, "    if a.device != b.device or a.shape != b.shape:"))
print(d)
