# scratch: why is the point coverage of the pieces (0.58) lower than the node recall (0.83)? one frame in detail
import gzip, os, pickle, sys
import numpy as np
from scipy.spatial import cKDTree
ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_eval as QE
store = "/mnt/c/Auto-research/data/autogt/lane/0702a065c6"
dump = pickle.load(open(os.path.join(ROOT, "temp/e276/predfull_e273_p_s0.pkl"), "rb"))
frames = QE.val_frames()[: len(dump)]
tot = []
for i in (0, 5, 50, 300, 600):
    f, d = frames[i], dump[i]
    g = pickle.load(gzip.open(os.path.join(store, f.token + ".pkl.gz"), "rb"))
    ref = QE.LaneRef(g)
    P = d["pos"][d["e"] >= 0.5]
    tree = cKDTree(P)
    allpts = np.concatenate([np.asarray(p["pts"], float) for p in g["pieces"]])
    gtn = ref.nt["pos"]
    dist_pts = tree.query(allpts)[0]
    dist_gtn = tree.query(gtn)[0]
    d_p_to_pts = cKDTree(allpts).query(P)[0]
    print(f"frame {i}: nodes {len(P)}, piece points {len(allpts)}, GT nodes {len(gtn)}; share of piece points with a node within 2 m {np.mean(dist_pts <= 2):.3f}; GT nodes with a node within 2 m {np.mean(dist_gtn <= 2):.3f}; "
          f"nodes with a piece point within 2 m {np.mean(d_p_to_pts <= 2):.3f}; piece pts x range {allpts[:,0].min():.0f}..{allpts[:,0].max():.0f}, y {allpts[:,1].min():.0f}..{allpts[:,1].max():.0f}; nodes x {P[:,0].min():.0f}..{P[:,0].max():.0f}")
    lens = [p["length"] for p in g["pieces"]]
    print("   pieces", len(lens), "total length %.0f m" % sum(lens), "mean pts spacing %.2f m" % (sum(lens) / max(len(allpts), 1)))
