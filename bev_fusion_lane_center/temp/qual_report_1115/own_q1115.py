"""scratch: own-lane (tier 1, visible, within 1.0 m) recall of the delivered graphs split ahead / behind the car and by range, rebuilt from the stored delivered graphs (pass1) and the lane GT; total checked against the counts pickles of the cards."""
import os, sys, pickle
import numpy as np
from scipy.spatial import cKDTree
R = os.environ["BFLC_ROOT"]; H = f"{R}/temp/qual_report_1115"
for sub in ("evaluation", "training/grid", "training", "training/data", "evaluation/sparse_lanes", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(R, sub)))
import qgraph_eval as QE
P1 = pickle.load(open(f"{H}/pass1.pkl", "rb"))
c324 = pickle.load(open(f"{R}/runs/e324_s_s0/ends_card_snap_thr0.1.counts.pkl", "rb"))
c312 = pickle.load(open(f"{R}/runs/e312_p_s0/ends_card_snap_thr0.1.counts.pkl", "rb"))
frames = QE.val_frames()
bins = (("behind (y<0)", -1e9, 0.0), ("ahead 0-20 m", 0.0, 20.0), ("ahead 20-40 m", 20.0, 40.0), ("ahead 40-48 m", 40.0, 48.0))
acc = {s: np.zeros((len(bins), 2)) for s in ("B", "E312", "E324s")}
nfr = {s: np.zeros(3) for s in acc}   # frames with ahead own lane (0-48 m) visible length >= 10 m, of them ahead recall < 0.8, < 0.5
for f, rec in zip(frames, P1["frames"]):
    ref = f.ref()
    G, t, v = ref.pts, np.where(ref.ptier > 0, ref.ptier, 3), ref.pvis
    inbox = (np.abs(G) < QE.BOX).all(1)
    m1 = inbox & (t == 1) & v
    for s in acc:
        dp = [QE._densify(l) for l in rec["sys"][s]["lines"] if len(l)]
        Pt = np.concatenate(dp) if dp else np.zeros((0, 2))
        Pt = Pt[(np.abs(Pt) < QE.BOX).all(1)]
        hit = (cKDTree(Pt).query(G)[0] <= 1.0) if len(Pt) else np.zeros(len(G), bool)
        for j, (nm, lo, hi) in enumerate(bins):
            mm = m1 & (G[:, 1] >= lo) & (G[:, 1] < hi)
            acc[s][j] += [(mm & hit).sum(), mm.sum()]
        ma = m1 & (G[:, 1] >= 0)
        if ma.sum() >= 20:                                      # >= 10 m of visible own lane ahead
            r = (ma & hit).sum() / ma.sum()
            nfr[s] += [1, r < 0.8, r < 0.5]
    f._lane = f._ref = None
for s in acc:
    tot = acc[s].sum(0)
    print(s, "total own-lane R(vis) %.4f" % (tot[0] / tot[1]), "| " + ", ".join(f"{nm} {a[0] / a[1]:.3f} ({a[1] / 2:.0f} m)" for (nm, _, _), a in zip(bins, acc[s])), "| frames with >=10 m own lane ahead: %d, recall<0.8: %d, <0.5: %d" % tuple(nfr[s]))
ref_tot = {"B": c324["base"], "E312": c312["head"], "E324s": c324["head"]}
for s, cs in ref_tot.items():
    r = sum(c["tier"]["rec"][0] for c in cs); print("card counts", s, round(r[0] / r[1], 4))
