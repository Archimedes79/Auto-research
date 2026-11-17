import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.environ["BFLC_ROOT"], "training", "grid"))
import test_lane_tokcache as T

m = [x for x in T.MUTANTS if x[1] == "the cache comparison ignores the tokens"][0]
with tempfile.TemporaryDirectory() as tmp:
    print(T.one((0, m, tmp)))
