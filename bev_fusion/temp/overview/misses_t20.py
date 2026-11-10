"""misses_t20.py - scratch (analysis, no verdict): which factors carry the MISSED lane_divider GT cells of t20 outside the map's intersection polygons?
Input: cached dt-0 maps (lane_readout.py run t20_base_s0), static GT arrays (readout_static.npz), the GT polylines (cached GT meta['vectors']['lane_divider']).
A GT cell / label point is MISSED when no predicted cell (p > 0.5) lies within 0.5 m of it (the metric's g_miss at tolerance 0.5 m).
  cell level  : all GT lane cells in the camera (fov) mask outside the intersection polygons, 0-60 m: range, occlusion (GT 'occluded' layer), box footprint
  point level : the 5-30 m label points of the paint check (rp_points.npz: stripe flags, tangents) outside intersections: range, |lateral position|, occluded, stripe / dash-gap candidate /
                isolated no stripe, distance to the nearest OTHER GT line (connected component of the GT lane mask), local curvature of the label line (heading change over +-1.5 m),
                scene light / rain (scene.json description)
  segment level: each GT polyline cut at the evaluated region's border (in the grid, fov, outside intersections) = a segment; found share per segment, runs of missed points along a line
Rates with scene-level bootstrap 95 % intervals (2000 resamples); a logistic model (IRLS) with a scene-cluster bootstrap (300 refits) for the odds ratios.
python temp/overview/misses_t20.py [--tag t20_base_s0__ckpt_00006000]   -> misses_t20_result.txt / .json"""
import argparse
import json
import os
import re
import sys

import numpy as np
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lane_readout as LR  # noqa: E402

sys.path.insert(0, os.path.join(LR.ROOT, "temp", "lane_tests"))
import paint_vs_label as PV  # noqa: E402


def logistic(X, y, iters=25, ridge=1e-4):
    """IRLS logistic regression (X with a constant column); returns the coefficients"""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = np.clip(X @ b, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        w = np.maximum(p * (1 - p), 1e-6)
        H = X.T @ (X * w[:, None]) + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p) - ridge * b)
        b = b + step
        if np.abs(step).max() < 1e-7:
            break
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="t20_base_s0__ckpt_00006000")
    a = ap.parse_args()
    frames = LR.load_frames()
    st = LR.get_static(frames)
    maps_dir = os.path.join(LR.MAPS_DIR, a.tag)
    done = [f for f in frames if os.path.exists(os.path.join(maps_dir, f"{f['i']}.npz"))]
    scene_ids = {s: k for k, s in enumerate(sorted({f["scene"] for f in frames}))}
    S = len(scene_ids)
    bt = LR.Boot(S)
    desc = {s["name"]: s["description"] for s in json.load(open(os.path.join(LR.FROOT, "v1.0-trainval", "scene.json")))}
    night = {n: bool(re.search(r"night", d, re.I)) for n, d in desc.items()}
    rain = {n: bool(re.search(r"rain|wet", d, re.I)) for n, d in desc.items()}
    L, R = [], {"tag": a.tag, "n_frames": len(done)}

    # ---------------------------------------------------------------- per-frame found maps, cell level, segments
    cell_rows = []          # (scene, x, y, found, occ, box) for the GT lane cells in fov outside intersections
    other_d = {}            # frame -> map of the distance to the nearest OTHER component (cells)
    found1, comp = {}, {}
    seg_rows, run_rows = [], []
    for f in done:
        i = f["i"]
        prob = np.load(os.path.join(maps_dir, f"{i}.npz"))["lane_divider"].astype(np.float32)
        P = prob > 0.5
        nP1 = LR.dilate(P, 1)
        found1[i] = nP1
        G, cam, I = st["lane"][i], st["cam"][i], st["inter"][i]
        m = G & cam & ~I
        ci, cj = np.nonzero(m)
        cell_rows.append(np.c_[np.full(len(ci), scene_ids[f["scene"]]), LR.XC[ci, cj], LR.YC[ci, cj], nP1[ci, cj], st["occ"][i][ci, cj], st["box"][i][ci, cj]])
        lab, n = ndi.label(G, structure=np.ones((3, 3)))
        d = np.full(G.shape, 99.0)
        for l in range(1, n + 1):
            own = lab == l
            oth = (lab > 0) & ~own
            if oth.any():
                d[own] = ndi.distance_transform_edt(~oth)[own] * LR.RES
        other_d[i] = d
        # segments / runs along the GT polylines
        g = np.load(os.path.join(LR.ROOT, "data", "cache", "gt", f["file"]), allow_pickle=True)["gt"].item()["cur"]
        for pl in g["meta"]["vectors"].get("lane_divider", []):
            pts, _ = PV.densify(pl)
            if len(pts) < 2:
                continue
            cx, cy = np.floor((pts[:, 0] - LR.X0) / LR.RES).astype(int), np.floor((pts[:, 1] - LR.Y0) / LR.RES).astype(int)
            ok = (cx >= 0) & (cx < LR.NX) & (cy >= 0) & (cy < LR.NY)
            ok[ok] = cam[cx[ok], cy[ok]] & ~I[cx[ok], cy[ok]]
            fl = np.zeros(len(pts), bool)
            fl[ok] = nP1[cx[ok], cy[ok]]
            idx = np.nonzero(ok)[0]
            if len(idx) == 0:
                continue
            brk = np.nonzero(np.diff(idx) > 1)[0] + 1
            for part in np.split(idx, brk):
                if len(part) < 2:
                    continue
                fr = fl[part].mean()
                seg_rows.append((scene_ids[f["scene"]], len(part) * 0.5, fr, float(pts[part[0], 0])))
                miss = ~fl[part]
                # runs of missed points
                edges = np.diff(np.r_[0, miss.astype(int), 0])
                for s0, e0 in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):
                    run_rows.append((scene_ids[f["scene"]], e0 - s0, len(part)))
    cells = np.concatenate(cell_rows)
    miss_c = 1 - cells[:, 3]

    def rate(sel, miss, scn, w=None):
        num = np.bincount(scn, miss * sel, S)
        den = np.bincount(scn, sel.astype(float), S)
        r = num.sum() / den.sum() if den.sum() else np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            v = (bt.W @ num) / (bt.W @ den)
        return r, LR.ci(v), int(den.sum())

    def table(title, rows_, miss, scn):
        L.append("")
        L.append(title)
        L.append(f"{'group':44s} {'n':>7s} {'missed [95% CI]':>22s}")
        for name, sel in rows_:
            r, c, n = rate(sel, miss, scn)
            L.append(f"{name:44s} {n:7d} {r:8.3f} [{c[0]:.3f}, {c[1]:.3f}]")
            R.setdefault(title[:40], {})[name] = dict(n=n, miss=r, ci=c)

    L.append(f"==== misses of t20 ({a.tag}): GT lane_divider cells / label points outside the map's intersection polygons that have NO predicted cell (p > 0.5) within 0.5 m ====")
    sc_c = cells[:, 0].astype(int)
    L.append(f"cell level: {len(cells)} GT lane cells in the fov mask outside intersections (0-60 m): missed {miss_c.mean():.3f}")
    xs = cells[:, 1]
    table("-- cell level, by range", [(f"{lo}-{hi} m", (xs >= lo) & (xs < hi)) for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60))], miss_c, sc_c)
    occ_c, box_c = cells[:, 4] > 0, cells[:, 5] > 0
    table("-- cell level, by occlusion / box footprint", [("not occluded, not in a box", ~occ_c & ~box_c), ("occluded (GT layer)", occ_c), ("inside a GT box footprint", box_c)], miss_c, sc_c)

    # ---------------------------------------------------------------- point level
    sel_all = np.array([f["i"] for f in done])
    pm = np.isin(st["pt_fi"], sel_all) & ~st["pt_inter"]
    ids = np.nonzero(pm)[0]
    fi = st["pt_fi"][ids]
    ci, cj = st["pt_ci"][ids], st["pt_cj"][ids]
    found = np.array([found1[f][a_, b_] for f, a_, b_ in zip(fi, ci, cj)])
    miss = 1.0 - found
    scn = np.array([scene_ids[frames[f]["scene"]] for f in fi])
    x, y, tx, ty = st["pt_x"][ids], st["pt_y"][ids], st["pt_tx"][ids], st["pt_ty"][ids]
    stripe, occ = st["pt_hit05"][ids], st["pt_occ"][ids]
    # dash-gap candidate
    near = np.zeros(len(ids), bool)
    for f in np.unique(fi):
        s_ = np.nonzero(fi == f)[0]
        xs_, ys_, txs, tys, hs = x[s_], y[s_], tx[s_], ty[s_], stripe[s_]
        dx, dy = xs_[None, :] - xs_[:, None], ys_[None, :] - ys_[:, None]
        al, la = dx * txs[:, None] + dy * tys[:, None], np.abs(-dx * tys[:, None] + dy * txs[:, None])
        near[s_] = ((np.abs(al) <= 6.0) & (la <= 0.75) & hs[None, :] & ~np.eye(len(s_), dtype=bool)).any(1)
    # curvature from the sequence of points (6 steps of 0.5 m)
    kappa = np.full(len(ids), np.nan)
    allx, ally = st["pt_x"], st["pt_y"]
    for q, gi in enumerate(ids):
        if gi < 3 or gi + 3 >= len(allx):
            continue
        j, k = gi - 3, gi + 3
        if not (st["pt_fi"][j] == st["pt_fi"][k] == st["pt_fi"][gi]):
            continue
        step = np.hypot(np.diff(allx[j:k + 1]), np.diff(ally[j:k + 1]))
        if (step < 0.15).any() or (step > 0.65).any():
            continue
        h = np.arctan2(st["pt_ty"][k], st["pt_tx"][k]) - np.arctan2(st["pt_ty"][j], st["pt_tx"][j])
        h = (h + np.pi) % (2 * np.pi) - np.pi
        kappa[q] = abs(h) / step.sum()
    od = np.array([other_d[f][a_, b_] for f, a_, b_ in zip(fi, ci, cj)])
    isnight = np.array([night[frames[f]["scene"]] for f in fi])
    israin = np.array([rain[frames[f]["scene"]] for f in fi])
    L.append("")
    L.append(f"point level: {len(ids)} label points (5-30 m, inside the image, outside intersections): missed {miss.mean():.3f}; stripe share {stripe.mean():.3f}; night scenes {isnight.mean():.3f} of the points, rain / wet {israin.mean():.3f}")
    table("-- points, by range", [(f"{lo}-{hi} m", (x >= lo) & (x < hi)) for lo, hi in ((5, 10), (10, 15), (15, 20), (20, 25), (25, 30))], miss, scn)
    ay = np.abs(y)
    table("-- points, by lateral position |y| (ego lane boundaries are about 1.7 m)", [(f"|y| {lo}-{hi} m", (ay >= lo) & (ay < hi)) for lo, hi in ((0, 2.5), (2.5, 5), (5, 8), (8, 12), (12, 30))], miss, scn)
    table("-- points, by occlusion (GT occluded layer) and paint", [("not occluded", ~occ), ("occluded", occ), ("stripe", stripe), ("no stripe", ~stripe), ("no stripe, a stripe of the line within 6 m (dash gap candidate)", ~stripe & near),
                                                                  ("no stripe, none within 6 m (isolated)", ~stripe & ~near), ("not occluded and stripe", ~occ & stripe), ("not occluded and isolated no stripe", ~occ & ~stripe & ~near)], miss, scn)
    table("-- points, by the distance to the nearest OTHER GT lane line (connected component of the GT mask; 7-100 m = no other line within 7 m or none in the frame)", [(f"{lo}-{hi} m", (od >= lo) & (od < hi)) for lo, hi in ((0, 1.5), (1.5, 3), (3, 4.5), (4.5, 7), (7, 100))], miss, scn)
    kk = np.nan_to_num(kappa, nan=-1)
    table("-- points, by the curvature of the label line (heading change over +-1.5 m / 3 m)", [("straight < 0.005 /m (R > 200 m)", (kk >= 0) & (kk < 0.005)), ("0.005 - 0.02 /m", (kk >= 0.005) & (kk < 0.02)), ("0.02 - 0.05 /m", (kk >= 0.02) & (kk < 0.05)),
                                                                                                         ("> 0.05 /m (R < 20 m)", kk >= 0.05), ("no curvature (end of a line)", kk < 0)], miss, scn)
    table("-- points, by scene light and weather (scene.json description)", [("day, dry", ~isnight & ~israin), ("night", isnight), ("rain / wet", israin), ("night and rain", isnight & israin)], miss, scn)
    # the 'most missed' scenes: is it scene-dominated?
    per_scene = np.bincount(scn, miss, S) / np.maximum(np.bincount(scn, minlength=S), 1)
    n_scene = np.bincount(scn, minlength=S)
    names = sorted(scene_ids, key=lambda s: scene_ids[s])
    used = np.nonzero(n_scene >= 30)[0]
    q = np.percentile(per_scene[used], [10, 25, 50, 75, 90])
    L += ["", f"-- scene level: {len(used)} scenes with >= 30 points; per-scene missed share: 10 / 25 / 50 / 75 / 90 % quantiles = " + " / ".join(f"{v:.2f}" for v in q)
          + f"; share of all misses carried by the worst 10 scenes: {np.sort(np.bincount(scn, miss, S))[::-1][:10].sum() / miss.sum():.2f}"]
    R["scene_quantiles"] = q.tolist()
    worst = np.argsort(-np.bincount(scn, miss, S))[:8]
    L.append("   the 8 scenes with the most missed points (name: points, missed share; description):")
    for s in worst:
        L.append(f"     {names[s]}: {int(n_scene[s])} points, missed {per_scene[s]:.2f}; {desc.get(names[s], '')[:90]}")
    L.append(f"   scenes of the 83 whose description contains 'night': {sum(night[s] for s in scene_ids)}; 'rain' / 'wet': {sum(rain[s] for s in scene_ids)} (the night column of the model is dropped when it has no variance)")

    # ---------------------------------------------------------------- segments and runs
    seg = np.array(seg_rows)
    L += ["", f"-- segment level: {len(seg)} GT line segments (polyline parts inside the evaluated region: grid, fov, outside intersections); found share of a segment's points (0.5 m tolerance)"]
    R["segments"] = {}
    for name, m in (("all", np.ones(len(seg), bool)), ("length < 5 m", seg[:, 1] < 5), ("5 - 15 m", (seg[:, 1] >= 5) & (seg[:, 1] < 15)), ("15 - 40 m", (seg[:, 1] >= 15) & (seg[:, 1] < 40)), ("> 40 m", seg[:, 1] >= 40)):
        s = seg[m]
        if len(s) == 0:
            continue
        sc_s = s[:, 0].astype(int)
        r0 = rate(np.ones(len(s), bool), (s[:, 2] == 0).astype(float), sc_s)
        r50 = rate(np.ones(len(s), bool), (s[:, 2] >= 0.5).astype(float), sc_s)
        rfull = rate(np.ones(len(s), bool), (s[:, 2] >= 0.95).astype(float), sc_s)
        pts_w = s[:, 1]
        L.append(f"  {name:14s} n {len(s):5d}  found nowhere {r0[0]:.3f} [{r0[1][0]:.3f}, {r0[1][1]:.3f}]   found >= 50 % {r50[0]:.3f} [{r50[1][0]:.3f}, {r50[1][1]:.3f}]   found >= 95 % {rfull[0]:.3f}   (length-weighted found share {np.average(s[:, 2], weights=pts_w):.3f})")
        R["segments"][name] = dict(n=len(s), nowhere=r0[0], ge50=r50[0], ge95=rfull[0])
    runs = np.array(run_rows)
    tot_miss_pts = runs[:, 1].sum()
    L += ["", f"-- runs of missed points along a line ({len(runs)} runs, {int(tot_miss_pts)} missed points of the polylines in the evaluated region; one point = 0.5 m of line): share of the missed points that sit in runs of"]
    R["runs"] = {}
    for lo, hi, nm in ((1, 3, "1-2 points (<= 1 m)"), (3, 7, "3-6 points (1.5 - 3 m)"), (7, 21, "7-20 points (3.5 - 10 m)"), (21, 61, "21-60 points (10 - 30 m)"), (61, 10000, "> 60 points (> 30 m)")):
        m = (runs[:, 1] >= lo) & (runs[:, 1] < hi)
        L.append(f"  {nm:28s} runs {int(m.sum()):6d}   missed points {runs[m, 1].sum() / tot_miss_pts:.3f}")
        R["runs"][nm] = dict(runs=int(m.sum()), share=float(runs[m, 1].sum() / tot_miss_pts))
    whole = runs[:, 1] >= 0.9 * runs[:, 2]
    L.append(f"  runs that cover >= 90 % of their segment (the whole line missed): {int(whole.sum())} runs, {runs[whole, 1].sum() / tot_miss_pts:.3f} of the missed points")
    R["runs"]["whole_segment_share"] = float(runs[whole, 1].sum() / tot_miss_pts)

    # ---------------------------------------------------------------- logistic model
    good = np.isfinite(kappa) & (od < 90)
    Xc = np.c_[np.ones(len(ids)), (x - 17.5) / 10.0, np.minimum(ay, 12) / 5.0, occ, ~stripe & near, ~stripe & ~near, np.minimum(od, 7.0) / 3.0, np.nan_to_num(kappa) / 0.02, isnight, israin].astype(float)
    cols = ["const", "range (per 10 m, from 17.5 m)", "|y| (per 5 m, capped 12 m)", "occluded", "no stripe, dash-gap candidate (vs stripe)", "no stripe, isolated (vs stripe)", "other-line distance (per 3 m, capped 7 m)",
            "curvature (per 0.02 /m)", "night", "rain / wet"]
    Xg, yg, sg = Xc[good], miss[good], scn[good]
    keepc = [0] + [j for j in range(1, Xg.shape[1]) if Xg[:, j].std() > 1e-9]
    cols = [cols[j] for j in keepc]
    Xg = Xg[:, keepc]
    b = logistic(Xg, yg)
    rng = np.random.RandomState(5)
    bs = []
    uniq = np.unique(sg)
    groups = {s: np.nonzero(sg == s)[0] for s in uniq}
    for _ in range(300):
        pick = rng.choice(uniq, len(uniq))
        ii = np.concatenate([groups[s] for s in pick])
        bs.append(logistic(Xg[ii], yg[ii]))
    bs = np.array(bs)
    L += ["", f"-- logistic model of 'missed' on {int(good.sum())} points (curvature and other-line distance defined; outside intersections, 5-30 m): odds ratios with a scene-cluster bootstrap (300 refits) 95 % interval",
          "   caveats: range, stripe, dash-gap and occlusion are correlated; the odds ratio of a column is conditional on the others; the stripe flag is a proxy; scene-level confounders (a few scenes carry many points) are only partly covered by the cluster bootstrap"]
    R["logistic"] = {}
    for j in range(1, len(cols)):
        lo, hi = np.exp(np.percentile(bs[:, j], [2.5, 97.5]))
        L.append(f"   {cols[j]:46s} OR {np.exp(b[j]):6.2f}  [{lo:.2f}, {hi:.2f}]")
        R["logistic"][cols[j]] = dict(OR=float(np.exp(b[j])), ci=[float(lo), float(hi)])
    from scipy.stats import rankdata
    p_hat = 1 / (1 + np.exp(-Xg @ b))
    rk = rankdata(p_hat)
    n1 = yg.sum()
    auc = (rk[yg > 0].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(yg) - n1))
    L.append(f"   in-sample AUC of the model for 'missed': {auc:.3f} (a measure of how much the listed factors separate missed from found points, not a validated predictor)")
    R["logistic_auc_insample"] = float(auc)
    txt = "\n".join(L) + "\n"
    open(os.path.join(HERE, "misses_t20_result.txt"), "w").write(txt)
    json.dump(R, open(os.path.join(HERE, "misses_t20_result.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(txt)


if __name__ == "__main__":
    main()
