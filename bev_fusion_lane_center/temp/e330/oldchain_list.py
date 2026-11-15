"""Old-chain candidates (scratch): files the live head does not import (kept alive only by the full test list, or by nobody), with line counts and the tests of the list that run them."""
import os
import re
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e285"))
import reach2  # noqa: E402

ROOT = reach2.ROOT
only_tests = sorted(reach2.alltests - reach2.active)
orph = sorted(reach2.allp - reach2.alltests - reach2.active)
skip = ("training/data/prep/", "tools/literature/")


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


with open(os.path.join(os.path.dirname(__file__), "oldchain.txt"), "w") as f:
    for tag, lst in (("T", only_tests), ("O", orph)):
        for p in lst:
            r = rel(p)
            if r.startswith(skip):
                continue
            f.write(f"{tag} {reach2.lines(p)} {r}\n")
# the lines of the test lists that name a file of only_tests
names = {os.path.basename(p) for p in only_tests}
with open(os.path.join(os.path.dirname(__file__), "oldchain_testlines.txt"), "w") as f:
    for sh in ("tools/maintenance/run_all_tests.sh", "tools/maintenance/run_gates.sh"):
        for i, ln in enumerate(open(os.path.join(ROOT, sh), encoding="utf-8"), 1):
            m = re.search(r'^run\s+"[^"]*"\s+(\S+\.py)', ln)
            if m and os.path.basename(m.group(1)) in names:
                f.write(f"{sh}:{i}: {ln.rstrip()}\n")
print("done")
