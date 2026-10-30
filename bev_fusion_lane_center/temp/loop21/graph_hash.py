"""Identity check of the assembly (E260): hash of all assembled graphs of VAL 873 (pts, edges, nodes) per registry name, written / compared.
    wsl bash tools/env/_py14.sh temp/loop21/graph_hash.py write|check [names ...]      (default: fast_recall champion recall fast_recall_na fast_recall_dp)
A hash file temp/loop21/graph_hash.json is written by `write` (before the code change) and compared by `check` (after)."""
import hashlib
import json
import os
import sys

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "loop20"))
import e255_matched as E  # noqa: E402
import p16_infer as PI  # noqa: E402

mode = sys.argv[1]
names = sys.argv[2:] or ["fast_recall", "champion", "recall", "fast_recall_na", "fast_recall_dp", "champion_dp"]
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph_hash.json")
stored = json.load(open(path)) if os.path.exists(path) else {}
bad = 0
for name in names:
    E.load(name)
    S = E.S
    h = hashlib.sha256()
    for r in S["raw"]:
        g = PI.assemble(r, None, S["tau"], S["cfg"], **S["asm"])
        for p, nd in zip(g["pts"], g["nodes"]):
            h.update(np.ascontiguousarray(np.asarray(p, np.float64)).tobytes())
            h.update(np.asarray(nd, np.int64).tobytes())
        h.update(np.asarray(sorted(map(tuple, g["edges"])), np.int64).tobytes())
    d = h.hexdigest()
    if mode == "write":
        stored[name] = d
        print(f"{name}: {d[:16]} written")
    else:
        ok = stored.get(name) == d
        bad += not ok
        print(f"{name}: {'IDENTICAL' if ok else 'DIFFERENT'} ({d[:16]} vs {str(stored.get(name))[:16]})")
if mode == "write":
    json.dump(stored, open(path, "w"), indent=1)
sys.exit(1 if bad else 0)
