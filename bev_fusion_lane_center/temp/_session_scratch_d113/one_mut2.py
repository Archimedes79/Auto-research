import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.environ["BFLC_ROOT"], "training", "grid"))
import test_lane_tokcache as T

with tempfile.TemporaryDirectory() as tmp:
    for k, m in enumerate([x for x in T.MUTANTS if "look-behind" in x[1] or "look behind" in x[1]]):
        print(T.one((k, m, tmp)))
