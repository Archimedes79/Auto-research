import os, sys, resource
import numpy as np
sys.path.insert(0, "training")
from feature_cache import FeatureStore
print("ulimit -n soft/hard", resource.getrlimit(resource.RLIMIT_NOFILE))
s = FeatureStore("data/cache/cam_feats_bevfusion")
n0 = len(os.listdir("/proc/self/fd"))
toks = list(s.index)[:: max(1, len(s.index) // 400)][:400]
for t in toks[:60]:
    s.get(t)
print("fds before", n0, "after", len(os.listdir("/proc/self/fd")), "shards opened", len(s._mm))
