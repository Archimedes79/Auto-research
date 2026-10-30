"""E258 step 0: where does PI.assemble spend its time? CPU, 1 thread, 200 cached frames of fast_recall at the registry settings (numba warm-up first).

    wsl bash tools/env/_py14.sh temp/loop21/e258_profile.py [name]
Prints: median ms per frame (wall), the load average, cProfile by own time and by cumulative time (top 18)."""
import cProfile
import os
import pstats
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "loop20"))
import numpy as np  # noqa: E402
import e255_matched as E  # noqa: E402
import p16_infer as PI  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "fast_recall"
E.load(name)
S = E.S
idx = np.linspace(0, len(S["raw"]) - 1, 200).astype(int)
raw = [S["raw"][i] for i in idx]


def run(rs):
    return [PI.assemble(r, None, S["tau"], S["cfg"], **S["asm"]) for r in rs]


run(raw[:20])  # numba compile / warm-up
print("load average", open("/proc/loadavg").read().strip())
ts = []
for r in raw:
    t = time.perf_counter()
    PI.assemble(r, None, S["tau"], S["cfg"], **S["asm"])
    ts.append((time.perf_counter() - t) * 1e3)
print(f"{name}: assemble wall ms per frame: median {np.median(ts):.2f}  mean {np.mean(ts):.2f}  p90 {np.percentile(ts, 90):.2f}  (200 frames)")
pr = cProfile.Profile()
pr.enable()
run(raw)
pr.disable()
st = pstats.Stats(pr)
tot = st.total_tt
print(f"profile total {tot * 1e3 / len(raw):.2f} ms per frame (with profiler overhead)")
for key in ("tottime", "cumulative"):
    print(f"--- by {key}")
    st.sort_stats(key).print_stats(18)
