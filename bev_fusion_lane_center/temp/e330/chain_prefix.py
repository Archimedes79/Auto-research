"""All tracked py/sh files of the old route-graph chain by name (p8 - p18 prefixes, route_data, chain files) with the verdict of the import closure: live head / test list / nobody (scratch)."""
import os
import re
import subprocess
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e285"))
import reach2  # noqa: E402

ROOT = reach2.ROOT
files = subprocess.run(["git", "ls-files", "--", "training", "evaluation", "tools"], cwd=ROOT, capture_output=True, text=True).stdout.split()
pat = re.compile(r"(^|/)(p(8|10|11|12|13|14|15|16|17|18|19)_[^/]*|route_data|dense_head|chain_data)\.(py|sh)$")
sel = [f for f in files if pat.search(f) and "/legacy/" not in f]
act = {os.path.relpath(p, ROOT).replace("\\", "/") for p in reach2.active}
tst = {os.path.relpath(p, ROOT).replace("\\", "/") for p in reach2.alltests - reach2.active}
out = []
for f in sorted(sel):
    st = "LIVE" if f in act else ("TESTLIST" if f in tst else "ORPHAN/none")
    n = sum(1 for _ in open(os.path.join(ROOT, f), encoding="utf-8", errors="replace"))
    out.append(f"{st:12s} {n:5d} {f}")
open(os.path.join(os.path.dirname(__file__), "chain_prefix.txt"), "w").write("\n".join(out) + "\n")
print(len(sel), "files")
