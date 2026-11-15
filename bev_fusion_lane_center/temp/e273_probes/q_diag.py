"""one-off: which GT lane / connector points does the node oracle not reproduce (real l3 frames)?"""
import sys, os, collections
sys.path.insert(0, "/mnt/c/Auto-research"); sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/training/grid")
import numpy as np
from scipy.spatial import cKDTree
import qtargets as QT
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/training/grid")
import test_qtargets as T
why = collections.Counter(); tot = collections.Counter()
for g in T.frames(120):
    nt = QT.node_table(g)
    dec = np.concatenate(QT.decode(nt)) if len(nt["src"]) else np.zeros((0, 2))
    tr = cKDTree(dec) if len(dec) else None
    nodes_by_piece = collections.defaultdict(list)
    for m in range(len(nt["pos"])):
        nodes_by_piece[int(nt["piece"][m])].append(int(nt["j"][m]))
    for pi, p in enumerate(g["pieces"]):
        q = np.asarray(p["pts"], float)
        kind = np.array(["lane"] * len(q), dtype=object)
        for s in p["segments"]:
            kind[s["i0"]: s["i1"] + 1] = s["kind"]
        miss = (tr.query(q)[0] > 0.3) if tr is not None else np.ones(len(q), bool)
        for k in ("lane", "connector"):
            tot[k] += int((kind == k).sum())
        for j in np.where(miss)[0]:
            k = kind[j]
            L = len(q)
            has_nodes = pi in nodes_by_piece
            first = min(nodes_by_piece[pi]) if has_nodes else None
            last = max(nodes_by_piece[pi]) if has_nodes else None
            if not has_nodes:
                r = "piece without any node"
            elif j < first:
                r = f"before the first node of the piece (start_open {p['start_open']})"
            elif j > last:
                r = f"after the last node of the piece (end_open {p['end_open']})"
            else:
                r = "between nodes"
            why[(k, r)] += 1
for (k, r), v in sorted(why.items(), key=lambda kv: -kv[1])[:12]:
    print(f"{k:9s} {v / tot[k]:.4f} of the {k} points  {r}")
