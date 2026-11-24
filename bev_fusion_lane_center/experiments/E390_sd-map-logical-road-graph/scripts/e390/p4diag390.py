"""E390 diagnostic (by hand, no verdict): where does P4 fail? the invariance test of `sdgraph_probe.invariance` on the VAL frames WITH and WITHOUT a node without geometry (the fallback anchors such a node at the
ego origin, which a rigid transform does not move) -> prints both. usage: python p4diag390.py <seed>"""
import os
import sys

import numpy as np

R = os.environ["BFLC_ROOT"]
for sub in ("training/data", "training/grid", "evaluation", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(R, sub)))
import lane_sdgraph_data as SG  # noqa: E402
import sdgraph_probe as SP  # noqa: E402

S = int(sys.argv[1]) if len(sys.argv) > 1 else 0
va = SG.load("val")
empty = [any(not [p for p in gs if len(p) >= 2] for gs in G["geo"]) for G in va["graphs"]]
enc, heads = SP.load_models("g", S)
idx_e = [k for k, e in enumerate(empty) if e]
idx_c = [k for k, e in enumerate(empty) if not e]
print(f"seed {S}: {len(idx_e)} of {len(empty)} VAL frames hold a node without geometry")
for name, idx in (("frames WITH an empty node", idx_e), ("frames WITHOUT", idx_c)):
    graphs = [va["graphs"][k] for k in idx]
    r = SP.invariance(enc, heads, graphs, draws=3)
    print(f"  {name}: {r}")
