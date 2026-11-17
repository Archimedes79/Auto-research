import os, sys, time, pickle
import numpy as np
R = "/mnt/c/Auto-research/bev_fusion_lane_center/"
os.environ.setdefault("BFLC_ROOT", R.rstrip("/"))
sys.path.insert(0, R + "evaluation")
import qends_eval as QEV   # sets sys.path for training etc.
import jwire as JW
import qgraph_eval as QE
import qgraph_readout as QR
seed = int(sys.argv[1]); nmax = int(sys.argv[2])
spec = QR.parse(QEV.DEFAULT_SPEC)
dump = pickle.load(open(R + f"temp/e276/predfull_e273_p_s{seed}.pkl", "rb"))
ends = {f["row"]: f for f in pickle.load(open(R + f"runs/e324_s_s{seed}/ends_val.pkl", "rb"))}
res = {m: [] for m in ("att", "kp")}
same = []   # for each pair-end: is the nearest wired piece of the attached end (cover by att pairs) the same piece (by coordinates) as for the kp (cover by kp pairs)
t0 = time.time()
for b, d in enumerate(dump[:nmax]):
    pred = QR.apply(dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                         links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"])), spec)
    graph, bk = QE.readout(pred, 0.5, 0.5)
    e = ends[d["row"]]
    k = e["kp_pos"].shape[0] // 2
    p = 1.0 / (1.0 + np.exp(-e["logit"])) * e["allowed"]
    ia, ib = np.nonzero(p >= 0.1)
    out = {}
    for m, att in (("att", True), ("kp", False)):
        pairs = QEV.make_pairs(e, ia, ib, k, attach=att)
        keep, pe = JW.kept_ends(graph.lines, bk["kind"], bk["piece_links"], np.asarray(pred["links"]["long"], bool), pairs, QEV.cover_pos())
        rows = []
        for pr in pairs:
            ci = JW.candidates(pe, pr[0], pr[1], True, QEV.WIRE_POS, QEV.WIRE_ANG)
            co = JW.candidates(pe, pr[2], pr[3], False, QEV.WIRE_POS, QEV.WIRE_ANG)
            rows.append(((ci[0][0], tuple(np.round(ci[0][2], 3))) if ci else None, (co[0][0], tuple(np.round(co[0][2], 3))) if co else None))
        out[m] = rows
    for ra, rk in zip(out["att"], out["kp"]):
        for j in (0, 1):
            res["att"].append(ra[j][0] if ra[j] else np.nan)
            res["kp"].append(rk[j][0] if rk[j] else np.nan)
            same.append((ra[j] is None and rk[j] is None) or (ra[j] is not None and rk[j] is not None and ra[j][1] == rk[j][1]))
    if (b + 1) % 100 == 0:
        print(b + 1, "frames", "%.0f s" % (time.time() - t0), flush=True)
A, K, S = np.array(res["att"]), np.array(res["kp"]), np.array(same)
print(f"seed {seed}: frames {min(nmax, len(dump))}, pair-ends {len(A)}")
for nm, X in (("attached", A), ("keypoint", K)):
    g = ~np.isnan(X)
    print(f"  {nm}: a piece end in reach (4 m/45 deg) {g.mean():.3f}; distance to the nearest such piece end: median {np.median(X[g]):.3f} mean {X[g].mean():.3f}; share<=0.5 {np.mean(X[g] <= .5):.3f} <=1 {np.mean(X[g] <= 1):.3f} <=2 {np.mean(X[g] <= 2):.3f}")
both = ~np.isnan(A) & ~np.isnan(K)
print(f"  both have a piece in reach: {both.mean():.3f}; median dist attached {np.median(A[both]):.3f} kp {np.median(K[both]):.3f}")
print(f"  only attached has one {np.mean(~np.isnan(A) & np.isnan(K)):.3f}; only kp {np.mean(np.isnan(A) & ~np.isnan(K)):.3f}")
print(f"  SAME wired piece for attached and kp (both None counts as same): {S.mean():.3f}; among those where either has a piece: {S[~np.isnan(A) | ~np.isnan(K)].mean():.3f}")
