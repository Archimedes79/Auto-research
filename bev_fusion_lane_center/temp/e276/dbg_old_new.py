"""scratch: node_table of the committed qtargets vs the E276 one on the 200-frame sample of test_qtargets: same nodes / dropped / links?"""
import importlib.util, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "training", "grid"))
import test_qtargets as TQ
import qtargets as NEW
spec = importlib.util.spec_from_file_location("qt_old", os.path.join(HERE, "qtargets_old_copy.py"))
OLD = importlib.util.module_from_spec(spec); spec.loader.exec_module(OLD)
tot = dict(n_old=0, n_new=0, d_old=0, d_new=0, diff=0)
for g in TQ.frames(200):
    if not any(len(p["pts"]) for p in g["pieces"]):
        continue
    a, b = OLD.node_table(g), NEW.node_table(g)
    tot["n_old"] += a["stats"]["nodes"]; tot["n_new"] += b["stats"]["nodes"]
    tot["d_old"] += a["stats"]["dropped"]; tot["d_new"] += b["stats"]["dropped"]
    tot["diff"] += int(not (np.array_equal(a["pos"], b["pos"]) and np.array_equal(a["src"], b["src"]) and np.array_equal(a["flags"], b["flags"])))
print(tot)
