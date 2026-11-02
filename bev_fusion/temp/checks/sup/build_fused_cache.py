"""throw-away: build the full-size item caches of tests/test_flow_fused.py (1500 train, 48 held-out) ahead of the GPU runs."""
import sys
import time

sys.path[:0] = ["tests"]
sys.argv = ["x"]
import test_flow_fused as t  # noqa: E402

t0 = time.time()
print(len(t.items(False, 1500)), len(t.items(True, 48)), f"{time.time() - t0:.0f} s")
