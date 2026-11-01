"""E259 step 2: assembly CPU time (time.process_time) per variant, interleaved A/B, 200 frames x 5 repetitions, 1 thread.
    wsl bash tools/env/_py14.sh temp/loop21/e259_cputime.py [registry name ...]    (default fast_recall champion)"""
import os
import sys
import time

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "loop20"))
import e255_matched as E  # noqa: E402
import p16_infer as PI  # noqa: E402

VARS = [("base", {}), ("anchor off", {"junc_thr": 0.0}), ("reloc off", {"reloc_thr": 0.0}), ("clean off", {"clean": False}), ("dp 0.2 stage", {"dp_eps": 0.2}), ("clean+anchor off", {"clean": False, "junc_thr": 0.0})]
REPS = 5
for name in sys.argv[1:] or ["fast_recall", "champion"]:
    E.load(name)
    S = E.S
    base = {k: v for k, v in S["asm"].items() if k != "dp_eps"}
    idx = np.linspace(0, len(S["raw"]) - 1, 200).astype(int)
    raw = [S["raw"][i] for i in idx]
    for r in raw[:20]:
        PI.assemble(r, None, S["tau"], S["cfg"], **base)
    cpu = {v[0]: [] for v in VARS}
    wall = {v[0]: [] for v in VARS}
    loads = []
    for rep in range(REPS):
        loads.append(open("/proc/loadavg").read().split()[0])
        order = VARS if rep % 2 == 0 else VARS[::-1]
        for lab, over in order:
            asm = dict(base)
            asm.update(over)
            c0, w0 = time.process_time(), time.perf_counter()
            for r in raw:
                PI.assemble(r, None, S["tau"], S["cfg"], **asm)
            cpu[lab].append((time.process_time() - c0) * 1e3 / len(raw))
            wall[lab].append((time.perf_counter() - w0) * 1e3 / len(raw))
    print(f"== {name}: mean ms per frame over 200 frames; 5 repetitions, load average at the repetitions {' '.join(loads)}")
    for lab, _ in VARS:
        d = np.array(cpu["base"]) - np.array(cpu[lab])
        print(f"  {lab:17} CPU {np.median(cpu[lab]):6.2f} ms [{min(cpu[lab]):.2f} - {max(cpu[lab]):.2f}]  wall {np.median(wall[lab]):6.2f}  saving vs base (CPU) per repetition " + " ".join(f"{x:+.2f}" for x in d) + f" -> median {np.median(d):+.2f}", flush=True)
