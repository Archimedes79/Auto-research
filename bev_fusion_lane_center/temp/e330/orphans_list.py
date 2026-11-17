"""List the orphan files (not imported by the live head, not run by a test) one per line, repo-relative to the lane folder (scratch)."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e285"))
import reach2  # noqa: E402

out = os.path.join(os.path.dirname(__file__), "orphans.txt")
with open(out, "w") as f:
    for p in sorted(reach2.allp - reach2.alltests - reach2.active):
        rel = os.path.relpath(p, reach2.ROOT).replace("\\", "/")
        if rel.startswith(("training/data/prep/", "tools/literature/")):
            continue
        f.write(rel + "\n")
print("written", out)
