import importlib.util, sys
import numpy as np
sys.path.insert(0, "evaluation/sparse_lanes")
import sparse_metrics as NEW
spec = importlib.util.spec_from_file_location("old_sm", sys.argv[1]); OLD = importlib.util.module_from_spec(spec); spec.loader.exec_module(OLD)
rng = np.random.default_rng(0)
bad = 0
for t in range(3000):
    n = int(rng.integers(2, 25)); m = int(rng.integers(1, 3 * n))
    xy = rng.normal(size=(n, 2))
    edges = [(int(rng.integers(0, n)), int(rng.integers(0, n))) for _ in range(m)]
    a, b = OLD.graph_from_vertices(xy, edges), NEW.graph_from_vertices(xy, edges, legacy_cut=True)
    if a["nodes"] != b["nodes"] or a["edges"] != b["edges"] or any(not np.array_equal(x, y) for x, y in zip(a["pts"], b["pts"])):
        bad += 1
print("graphs:", 3000, "differ:", bad)
