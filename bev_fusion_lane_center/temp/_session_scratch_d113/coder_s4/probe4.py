import time, os, sys
t0 = time.time()
import numpy, torch
print("numpy+torch", time.time() - t0)
ROOT = os.environ["BFLC_ROOT"]
for sub_ in ("training/data", "training/grid", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub_))
t0 = time.time(); import lane_tok_data; print("LT", time.time() - t0)
t0 = time.time(); import lane_tokcache, lane_ptok_data, qgraph, qpredict, jtoken, lane_past_data; print("rest", time.time() - t0)
