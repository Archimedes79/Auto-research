"""U3-U5 scratch: the run folders under runs/ whose config.json (design jtoken or jlive) holds a removed option (D117, the final REMOVED of the worktree's jtoken_train) at another value than its default:
the worktree's code refuses to load or resume them. Reads config.json only."""
import glob
import json
import os
import sys

import jtoken_train as JKT

ROOT = os.environ["BFLC_ROOT"]
rows, n = [], 0
for p in sorted(glob.glob(os.path.join(ROOT, "runs", "*", "config.json"))):
    try:
        c = json.load(open(p))
    except (OSError, ValueError):
        continue
    if c.get("design") not in ("jtoken", "jlive"):
        continue
    n += 1
    a = c.get("args") or {}
    held = {k: a[k] for k, v in JKT.REMOVED.items() if a.get(k) not in (None, v)}
    if held:
        rows.append((os.path.basename(os.path.dirname(p)), c["design"], held))
print(f"REMOVED = {JKT.REMOVED}; {n} run folders of design jtoken / jlive, {len(rows)} hold a removed option at another value:")
for name, design, held in rows:
    print(f"  {name:<28} {design:<7} {held}")
print("scan done", file=sys.stderr)
