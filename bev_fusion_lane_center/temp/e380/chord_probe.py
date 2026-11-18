"""E380 step 0 (scratch, CPU): the chord probe of card E380. Does the evidence of B''s frozen pyramid maps along the straight chord of a token pair add to the token existence + the 13 pair-geometry features when the question is 'is this long pair a GT junction link'?
VAL frames of the cache of B' seed 0 (the live pyramid of `runs/e292_a_s0` on the CPU, B''s own x of the frame); pairs = ASSIGNED tokens i, j (both matched to a GT node) inside the allowed gate of `jtok.pair_geometry` with a chord above 8 m (the long class); label = a GT LONG link between the two nodes
(the table's src / dst / long); all positives and up to NEG random negatives per frame. Features: baseline = the 13 pair-geometry features + the two existence logits; evidence = eight numbers from five bilinear samples at t = 1/6 .. 5/6 of the chord: mean and min of the 1 m centre-lane probability (centre1 ch 0),
mean road and mean intersection probability (road2 ch 0 / 1), max stop and max exit heat (centre1 ch 8 / 9), mean lane-divider and mean road-boundary probability (lines05 ch 0 / 2). A HistGradientBoostingClassifier (the same settings for both feature sets), 5 contiguous folds of VAL frames (fit on four, test on the fifth); AUC and AP per fold,
the difference. In-sample on VAL, a probe, never a verdict (the pairs are of GT-assigned tokens: the wiring question without the existence question). usage: python chord_probe.py [--n 873] [--neg 400] [--out temp/e380/chord_probe.json]"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.ndimage import map_coordinates
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

R = os.environ["BFLC_ROOT"]
for sub in ("training/data", "training/grid", "evaluation", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(R, sub)))
import jtok as JT  # noqa: E402
import lane_live_tok as LL  # noqa: E402
import lane_ptok_data as LP  # noqa: E402
import lane_tokcache as LC  # noqa: E402
import qpredict as QP  # noqa: E402
import qtargets as QT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=873)
ap.add_argument("--neg", type=int, default=400)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="temp/e380/chord_probe.json")
a = ap.parse_args()
torch.set_num_threads(2)
rng = np.random.default_rng(0)
ds = LL.LiveData("val", a.seed)
c = LC.TokCache(LP.cache_dir("val", a.seed, "e292_a"))
tok2i = {t: i for i, t in enumerate(c.tokens)}
model = QP.load_model(os.path.join(R, "runs", f"e292_a_s{a.seed}"), "cpu").eval()
TS = np.array([1, 2, 3, 4, 5]) / 6.0
SHORT = QT.SHORT


def sample(arr, res, pts):
    """bilinear samples of one (H, W) map (rows = x, columns = y, origin -50 m, cell centres at +0.5) at the points pts (N, 2) in metres -> (N,)"""
    u = (pts[:, 0] + 50.0) / res - 0.5
    v = (pts[:, 1] + 50.0) / res - 0.5
    return map_coordinates(arr, [u, v], order=1, mode="nearest")


sig = lambda x: 1.0 / (1.0 + np.exp(-x))  # noqa: E731
X, Y, FR, t0 = [], [], [], time.time()
for fi in range(a.n):
    item = ds[fi]
    i = tok2i[item["token"]]
    with torch.no_grad():
        P = model.pyr(item["x"][None].float())
    cen, road, lin = (P[k][0].numpy() for k in ("centre1", "road2", "lines05"))
    pos, dirv = np.array(c.f["pos"][i], np.float64), np.array(c.f["dirv"][i], np.float64)
    e, valid = np.array(c.f["e"][i], np.float64), np.array(c.f["valid"][i], bool)
    tk = np.array(c.f["tok"][i], np.float32)
    gid = np.array(c.f["gt_id"][i])
    tb = item["table"]
    gl = set((int(s), int(d)) for s, d, lg in zip(tb["src"], tb["dst"], tb["long"]) if lg)
    pt = torch.from_numpy(pos)[None].float()
    g, allowed = JT.pair_geometry(pt, torch.from_numpy(dirv)[None].float(), torch.from_numpy(valid)[None], e=torch.from_numpy(sig(e))[None].float())
    g, allowed = g[0].numpy(), allowed[0].numpy()
    chord = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    asg = valid & (gid >= 0)
    cand = allowed & asg[:, None] & asg[None, :] & (chord > SHORT)
    ii, jj = np.nonzero(cand)
    if len(ii) == 0:
        continue
    lab = np.array([(int(gid[x]), int(gid[y])) in gl for x, y in zip(ii, jj)])
    keep = lab | (rng.random(len(ii)) < min(1.0, a.neg / max(int((~lab).sum()), 1)))
    ii, jj, lab = ii[keep], jj[keep], lab[keep]
    ev = np.zeros((len(ii), 8))
    for k, (x, y) in enumerate(zip(ii, jj)):
        pts = pos[x][None] * (1 - TS[:, None]) + pos[y][None] * TS[:, None]
        pc = sig(sample(cen[0], 1.0, pts))
        ev[k] = [pc.mean(), pc.min(), sig(sample(road[0], 2.0, pts)).mean(), sig(sample(road[1], 2.0, pts)).mean(), sig(sample(cen[8], 1.0, pts)).max(), sig(sample(cen[9], 1.0, pts)).max(),
                 sig(sample(lin[0], 0.5, pts)).mean(), sig(sample(lin[2], 0.5, pts)).mean()]
    base = np.concatenate([g[ii, jj], e[ii][:, None], e[jj][:, None]], 1)
    X.append(np.concatenate([base, tk[ii], tk[jj], ev], 1))
    Y.append(lab)
    FR.append(np.full(len(ii), fi))
    if fi % 50 == 0:
        print(f"frame {fi}/{a.n}: {len(ii)} pairs ({int(lab.sum())} positive), {time.time() - t0:.0f} s", flush=True)
X, Y, FR = np.concatenate(X), np.concatenate(Y), np.concatenate(FR)
nb = X.shape[1] - 8
ng = nb - 128                                      # the geometry block: 13 pair features + 2 existence logits; then the two tokens' 64-channel node-head features, then the 8 evidence numbers
print(f"pairs {len(Y)}, positives {int(Y.sum())} ({Y.mean():.3f}), features: baseline {nb}, evidence 8")
folds = np.minimum(FR * 5 // a.n, 4)
res = []
for f in range(5):
    tr, te = folds != f, folds == f
    out = {}
    G, T, E = np.arange(0, ng), np.arange(ng, nb), np.arange(nb, nb + 8)
    for name, cols in (("geo", G), ("geo+ev", np.r_[G, E]), ("geo+tok", np.r_[G, T]), ("geo+tok+ev", np.r_[G, T, E]), ("ev_only", E)):
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0).fit(X[tr][:, cols], Y[tr])
        s = m.predict_proba(X[te][:, cols])[:, 1]
        out[name] = (roc_auc_score(Y[te], s), average_precision_score(Y[te], s))
    res.append(dict(fold=f, pairs=int(te.sum()), positives=int(Y[te].sum()), **{k: dict(auc=v[0], ap=v[1]) for k, v in out.items()}))
    print("fold %d: pairs %d pos %d | AUC geo %.4f geo+ev %.4f (%+.4f) | geo+tok %.4f geo+tok+ev %.4f (%+.4f) | ev only %.4f | AP geo+tok %.4f geo+tok+ev %.4f (%+.4f)" % (f, te.sum(), Y[te].sum(), out["geo"][0], out["geo+ev"][0], out["geo+ev"][0] - out["geo"][0], out["geo+tok"][0], out["geo+tok+ev"][0], out["geo+tok+ev"][0] - out["geo+tok"][0], out["ev_only"][0], out["geo+tok"][1], out["geo+tok+ev"][1], out["geo+tok+ev"][1] - out["geo+tok"][1]), flush=True)
g1 = [r["geo+ev"]["auc"] - r["geo"]["auc"] for r in res]
g2 = [r["geo+tok+ev"]["auc"] - r["geo+tok"]["auc"] for r in res]
print("AUC gain of the evidence per fold: over geometry only", " ".join("%+.4f" % x for x in g1), "| mean %+.4f | folds >= 0.03: %d of 5" % (np.mean(g1), sum(x >= 0.03 for x in g1)))
print("AUC gain of the evidence per fold: over geometry + the tokens' node-head features (the stronger baseline, the card's criterion)", " ".join("%+.4f" % x for x in g2), "| mean %+.4f | folds >= 0.03: %d of 5" % (np.mean(g2), sum(x >= 0.03 for x in g2)))
json.dump(dict(args=vars(a), pairs=int(len(Y)), positives=int(Y.sum()), folds=res, auc_gain_geo=g1, auc_gain_tok=g2), open(os.path.join(R, a.out), "w"), indent=1)
