"""scratch: which frame of the test_qtargets sample has no GT lane points?"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "training", "grid"))
import test_qtargets as TQ  # noqa: E402
import qtargets as QT  # noqa: E402

bad = 0
for g in TQ.frames(int(sys.argv[1]) if len(sys.argv) > 1 else 60):
    nt = QT.node_table(g)
    npts = sum(len(p["pts"]) for p in g["pieces"])
    kinds = {p["kind"] for p in g["pieces"]}
    if npts == 0 or not len(nt["pos"]):
        bad += 1
        print("empty frame", g["sample"], "pieces", len(g["pieces"]), "points", npts, "nodes", len(nt["pos"]), "location", g.get("location"))
print("frames without lane points:", bad)
