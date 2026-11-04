"""throw-away: how large are the GPU differences between two runs of the same command, and is door-vs-suite inside that noise?"""
import json
import math
import os
import sys

R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "door_check"))
load = lambda w: json.load(open(os.path.join(R, w, "t16_full_s0", "eval", "metrics.json")))["metrics"]  # noqa: E731
for x, y in (a.split(":") for a in sys.argv[1:]):
    ma, mb = load(x), load(y)
    d = {k: abs(ma[k] - mb[k]) for k in ma if isinstance(ma[k], float) and isinstance(mb.get(k), float) and ma[k] == ma[k] and mb[k] == mb[k] and ma[k] != mb[k]}
    nan_mismatch = [k for k in ma if isinstance(ma[k], float) and (ma[k] != ma[k]) != (mb.get(k) != mb.get(k))]
    rel = sorted(v / max(abs(ma[k]), 1e-9) for k, v in d.items())
    print(f"{x} vs {y}: {len(d)} floats differ, nan mismatch {len(nan_mismatch)}, median rel {rel[len(rel) // 2]:.1e}, p95 rel {rel[int(len(rel) * .95)]:.1e}")
    for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:4]:
        print(f"   {k}: {ma[k]:.6g} vs {mb[k]:.6g}")
