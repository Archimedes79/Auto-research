"""Loop 21 one-off: keys and shapes of one assembled graph (cached raw outputs, fast_recall registry settings)."""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "loop20"))
import numpy as np  # noqa: E402
import e255_matched as E  # noqa: E402
import p16_infer as PI  # noqa: E402

E.load("fast_recall")
S = E.S
g = PI.assemble(S["raw"][5], None, S["tau"], S["cfg"], **S["asm"])
print("keys", sorted(g.keys()))
for k, v in g.items():
    if isinstance(v, (list, tuple)):
        print(k, "list", len(v), type(v[0]).__name__ if v else "", (np.asarray(v[0]).shape if v and hasattr(v[0], "__len__") else v[:3] if v else ""))
    elif hasattr(v, "shape"):
        print(k, "array", v.shape)
    else:
        print(k, type(v).__name__, str(v)[:80])
print("nodes[0]", g["nodes"][0] if g.get("nodes") else None)
print("pts[0] first rows", np.asarray(g["pts"][0])[:5])
