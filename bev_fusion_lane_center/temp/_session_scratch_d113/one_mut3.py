import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.environ["BFLC_ROOT"], "training", "grid"))
import test_lane_tokcache as T

ms = [x for x in T.MUTANTS if x[0] == T.PT and "ylong" in x[1]]
with tempfile.TemporaryDirectory() as tmp:
    for k, m in enumerate(ms):
        print(T.one((k, m, tmp)))
