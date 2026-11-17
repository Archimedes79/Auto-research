"""throw-away: the E355 toy checks alone (no real frames)"""
import sys
import time

import e346_arm as A

t = time.time()
A.check_merge_toys()
print("merge toys ok", round(time.time() - t, 1), flush=True)
A.check_e355_variants((2.0, 1.0, 2.5, 3.125))
print("variant toys ok", round(time.time() - t, 1), flush=True)
if len(sys.argv) > 1:
    A.check_variant(sys.argv[1])
    print("variant", sys.argv[1], "ok", round(time.time() - t, 1), flush=True)
