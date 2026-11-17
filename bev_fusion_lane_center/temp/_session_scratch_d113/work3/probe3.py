import os, sys
import numpy as np
R = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(R, "training/data"))
import lane_map_data as L
sizes = [len(L.error_map(L.toy_pts(0), f"t{k}")[0]) for k in range(300)]
from collections import Counter
print(Counter(sizes))
