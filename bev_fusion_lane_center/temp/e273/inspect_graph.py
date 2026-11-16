import os, sys, numpy as np
R = os.environ["BFLC_ROOT"]
for sub in ("evaluation", "training", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(R, sub)))
import lane_suite as LS, qgraph_eval as QE
sub = LS.GraphSubject("recall_sp0_dp"); fr = QE.val_frames(); g = sub.graphs[fr[150].row]
print(type(g), list(g.keys()) if hasattr(g, "keys") else "")
for k, v in g.items():
    if isinstance(v, (list, tuple)):
        print(k, "list", len(v), type(v[0]), (np.asarray(v[0]).shape if len(v) else None))
    elif hasattr(v, "shape"):
        print(k, "array", v.shape, v.dtype)
    else:
        print(k, type(v), str(v)[:80])
