"""Loop 21 probe (one-off, CPU, 1 thread): how many of the vertices of the assembled graphs does Douglas-Peucker (0.3 m, the rule of the oracle 14.3) keep?
    wsl bash tools/env/_py14.sh temp/loop21/probe_dp.py <registry name> [...]
Assembles the cached raw outputs of VAL 873 at the registry settings; prints vert100 (node ids) next to vert100_dp (eps 0.3, and unique_ends) and Q."""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "loop20"))
import e255_matched as E  # noqa: E402
import p16_infer as PI  # noqa: E402
import sparse_metrics as SM  # noqa: E402

for name in sys.argv[1:]:
    E.load(name)
    S = E.S
    g = [PI.assemble(r, None, S["tau"], S["cfg"], **S["asm"]) for r in S["raw"]]
    m = SM.card_any(g, S["gts"], extra=True)
    print(f"== {name}: Q {m['Q']:.4f} vert100 (nodes) {m['vert100']:.2f} | vert100_dp 0.3 m {SM.vert100_dp(g, 0.3):.2f} unique_ends {SM.vert100_dp(g, 0.3, unique_ends=True):.2f} | "
          f"dp 0.15 {SM.vert100_dp(g, 0.15, unique_ends=True):.2f} dp 0.5 {SM.vert100_dp(g, 0.5, unique_ends=True):.2f}", flush=True)
