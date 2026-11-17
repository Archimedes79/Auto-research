import sys, dataclasses
sys.path[:0] = ["training/lanegraph", "training/lanegraph/scripts", "training/lanegraph/tests"]
import numpy as np
import test_lg_equivalence as TE
from lg_config import CHAMPION
import lg_readout as R
orig = R.deliver
st = dict(frames=0, conn=0, wired_in=0, wired_out=0, links=0, long=0, covered=0)
def spy(lanes, pairs, cfg):
    g = orig(lanes, pairs, cfg)
    if not cfg.attached_ends:
        pieces, _ = R.lane_pieces(lanes, cfg)
        st["frames"] += 1; st["links"] += sum(p[2] == "link" for p in pieces); st["long"] += sum(p[3] for p in pieces)
        st["covered"] += len(pieces) - sum(k != "connector" for k in g.kinds)
        m0 = sum(k != "connector" for k in g.kinds)
        for m in range(m0, len(g.kinds)):
            st["conn"] += 1; st["wired_in"] += any(e[1] == m for e in g.edges); st["wired_out"] += any(e[0] == m for e in g.edges)
    return g
TE.deliver = spy
print("diffs", TE.fuzz(300), st)
