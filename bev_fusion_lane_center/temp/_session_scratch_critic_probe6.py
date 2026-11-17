import os, sys, numpy as np
R = os.environ["BFLC_ROOT"]
for sub in ("training/data", "training/grid", "evaluation", ".."):
    sys.path.insert(0, os.path.join(R, sub))
import lane_map_data as LMD, lane_gt_data as LD, lane_tok_data as LT
from scipy.spatial import cKDTree
ds = LMD.MapData("val", 0, "e292_a", None, "exact")
td = LT.TokData("val", res=2.0, feats=False)
assert td.tokens == ds.tokens
lg, _ = LD.lane_store()
idx = np.linspace(0, len(ds) - 1, 60).astype(int)
tot = on = 0; noconn_sup = all_sup = 0; n_nodes = 0
for i in idx:
    g = lg.load(ds.tokens[i])
    t = td.table(int(i)); gp = t["pos"]
    if len(gp) == 0: continue
    conn = []; lane = []
    for p in g["pieces"]:
        pts = np.asarray(p["pts"], float)
        for s in p["segments"]:
            q = pts[s["i0"]:s["i1"] + 1]
            (conn if (s["kind"] == "connector" and s["in_intersection"]) else lane).append(q)
    if not conn: 
        tot += len(gp); continue
    C = cKDTree(np.concatenate(conn)); L = cKDTree(np.concatenate(lane)) if lane else None
    dc = C.query(gp)[0]; dl = L.query(gp)[0] if L is not None else np.full(len(gp), 99.)
    on += int(((dc < 1.0) & (dl >= 1.0)).sum()); tot += len(gp)
print(f"GT nodes on in-intersection connectors only (within 1 m of one, not within 1 m of any non-connector lane point): {on} of {tot} = {on/tot:.3f}")
