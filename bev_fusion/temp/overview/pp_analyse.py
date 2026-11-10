"""scratch (analysis, no verdict): tables of the precision-side test from temp/overview/precision_paint_cells.npz (pp_precision.py). Writes precision_paint_result.txt / .json.
python temp/overview/pp_analyse.py"""
import json
import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RANGES = ((5.0, 12.0), (12.0, 20.0), (20.0, 30.0), (5.0, 30.0))
NB = 2000
rng = np.random.RandomState(0)


def main():
    d = np.load(os.path.join(HERE, "precision_paint_cells.npz"))
    aux = pickle.load(open(os.path.join(HERE, "pp_aux.pkl"), "rb"))
    FEATS = [str(s) for s in d["FEATS"]]
    x, dist, scene = d["x"], d["dist"], d["scene"].astype(int)
    N, S = len(x), int(scene.max()) + 1
    W = rng.multinomial(S, np.ones(S) / S, size=NB).astype(float)                    # scene resample weights, the same for every interval
    TRUE = dist <= 0.5 + 1e-9
    NEAR = (~TRUE) & (dist <= 1.0 + 1e-9)
    FAR = dist > 1.0 + 1e-9
    FALSE = ~TRUE
    clear, okr, occ, box = d["clear"].astype(bool), d["okr"].astype(bool), d["occ"].astype(bool), d["box"].astype(bool)
    F = {n: d["feat"][:, k].astype(bool) for k, n in enumerate(FEATS)}
    R = {"n_cells": int(N), "n_scenes": int(len(np.unique(scene)))}
    L = []

    def rate(num, den, group=None):
        """pooled rate and scene-bootstrap 95 % interval of sum(num)/sum(den)"""
        a, b = np.bincount(scene, num.astype(float), S), np.bincount(scene, den.astype(float), S)
        r = a.sum() / b.sum() if b.sum() else np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            v = (W @ a) / (W @ b)
        ci = tuple(np.nanpercentile(v, [2.5, 97.5])) if np.isfinite(v).sum() > 100 else (np.nan, np.nan)
        return r, ci, int(b.sum()), a, b

    def diff(ra, rb):
        """a - b with an interval from the same resamples"""
        with np.errstate(divide="ignore", invalid="ignore"):
            v = (W @ ra[3]) / (W @ ra[4]) - (W @ rb[3]) / (W @ rb[4])
        return ra[0] - rb[0], tuple(np.nanpercentile(v, [2.5, 97.5]))

    fmt = lambda r: f"{r[0]:.3f} [{r[1][0]:.3f}, {r[1][1]:.3f}]"          # noqa: E731
    # ---- sanity: the precision of the metric's own logic
    p = aux["prec"]
    L += ["==== sanity: precision (predicted cells with p > 0.5 inside the camera (fov) mask that have a GT lane_divider cell within 0.5 m), this code vs the metric ====",
          f"all fov cells 0-60 m, 246 frames: {p['all'][0] / p['all'][1]:.3f} ({int(p['all'][1])} cells)   (metric counts of the same frames: 0.645)"
          + "   by metric range bin: " + ", ".join(f"{k} m {p[k][0] / p[k][1]:.3f}" for k in ("0-20", "20-40", "40-60")),
          f"analysis set (5-30 m, fov mask, centre projects into the image): {N} predicted cells in {R['n_scenes']} scenes; TRUE {TRUE.mean():.3f} (= precision of this set), FALSE {FALSE.mean():.3f} (near 0.5-1.0 m: {NEAR.mean():.3f}, far > 1.0 m: {FAR.mean():.3f})"]
    R["precision_all_fov"] = float(p["all"][0] / p["all"][1])
    R["precision_set"] = float(TRUE.mean())
    for lo, hi in RANGES:
        m = (x >= lo) & (x < hi)
        r = rate(TRUE & m, m)
        L.append(f"   precision {lo:.0f}-{hi:.0f} m: {fmt(r)}  (n {r[2]}); FALSE-near {NEAR[m].mean():.3f}, FALSE-far {FAR[m].mean():.3f}")
        R.setdefault("precision_by_range", {})[f"{lo:.0f}-{hi:.0f}"] = dict(precision=r[0], ci=r[1], n=r[2], near=float(NEAR[m].mean()), far=float(FAR[m].mean()))
    L.append(f"   cells with a clear line axis (>= 4 predicted cells within 1.5 m, eigenvalue ratio >= 3): {clear.mean():.3f}; whose full +-1 m profile lies inside the image: {okr.mean():.3f}; both: {(clear & okr).mean():.3f}")
    L.append(f"   distance of the FALSE cells to the nearest GT lane_divider cell: " + ", ".join(f"<= {t} m: {(dist[FALSE] <= t + 1e-9).mean():.3f}" for t in (0.75, 1.0, 1.5, 2.0, 3.0, 5.0)) + f"; median {np.median(dist[FALSE]):.2f} m")
    R["false_dist_cdf"] = {str(t): float((dist[FALSE] <= t + 1e-9).mean()) for t in (0.75, 1.0, 1.5, 2.0, 3.0, 5.0)}

    # ---- 1. stripe hit rates
    variants = (("clear axis, all cells (main)", clear & okr), ("all cells (unclear axis -> ego-forward axis)", okr), ("clear axis, not occluded and not in a box footprint", clear & okr & ~occ & ~box))
    R["stripe"] = {}
    for vname, vm in variants:
        L += ["", f"==== 1. stripe at the predicted cell, {vname}: stripe = contrast >= 0.25 within +-0.5 m (hit05) / +-0.2 m (hit02) across the line; control = the same cells moved {1.75} m sideways ====",
              f"{'range':8s} {'group':11s} {'n':>6s} {'stripe +-0.5 [95% CI]':>24s} {'+-0.2':>7s} | {'control +-0.5':>13s} {'n ctl':>6s} | {'vs TRUE: diff +-0.5 [95% CI]':>31s}"]
        for lo, hi in RANGES:
            rm = (x >= lo) & (x < hi) & vm
            res = {}
            for gname, gm in (("TRUE", TRUE), ("FALSE", FALSE), ("FALSE-near", NEAR), ("FALSE-far", FAR)):
                m = rm & gm
                r5 = rate(d["hit05"] & m, m)
                r2 = rate(d["hit02"] & m, m)
                cok, ch = d["c_ok"].astype(bool), d["c_hit05"].astype(bool)
                nc = np.bincount(scene, (ch & m[None]).sum(0).astype(float), S)
                dc = np.bincount(scene, (cok & m[None]).sum(0).astype(float), S)
                rc = (nc.sum() / dc.sum() if dc.sum() else np.nan, None, int(dc.sum()), nc, dc)
                res[gname] = (r5, r2, rc)
            for gname, (r5, r2, rc) in res.items():
                if gname == "TRUE":
                    dtxt = ""
                else:
                    dd = diff(r5, res["TRUE"][0])
                    dtxt = f"{dd[0]:+.3f} [{dd[1][0]:+.3f}, {dd[1][1]:+.3f}]"
                L.append(f"{lo:3.0f}-{hi:<4.0f} {gname:11s} {r5[2]:6d} {fmt(r5):>24s} {r2[0]:7.3f} | {rc[0]:13.3f} {rc[2]:6d} | {dtxt:>31s}")
                R["stripe"].setdefault(vname, {}).setdefault(f"{lo:.0f}-{hi:.0f}", {})[gname] = dict(n=r5[2], hit05=r5[0], ci05=r5[1], hit02=r2[0], control05=rc[0], n_control=rc[2],
                                                                                                       diff_vs_true=(None if gname == "TRUE" else dd))
    # ---- 2. position against the other GT layers
    base, gtl = aux["base"], aux["gtl"]
    names = FEATS + ["box", "occluded", "drivable", "walkway", "carpark"]
    bshare = dict(zip(names, base["feat"] / base["n"]))
    gshare = dict(zip(names, gtl["feat"] / gtl["n"]))
    rows = [("road_divider within 0.5 m", "rd05"), ("road_divider within 1.0 m", "rd10"), ("ped_crossing within 0.5 m", "pc05"), ("ped_crossing within 1.0 m", "pc10"),
            ("stop_line within 0.5 m", "sl05"), ("stop_line within 1.0 m", "sl10"), ("walkway boundary within 0.5 m", "wb05"), ("walkway boundary within 1.0 m", "wb10"),
            ("GT centreline within 0.5 m", "cl05"), ("GT centreline within 1.0 m", "cl10"),
            ("any of road_div/ped_cross/stop_line/walkway edge within 1.0 m", "oth10"), ("... or GT centreline within 1.0 m", "oth_cl10")]
    other = [("inside a GT box footprint (instance >= 0)", box, "box"), ("marked occluded in the GT", occ, "occluded"), ("inside the drivable area", d["dr"].astype(bool), "drivable"),
             ("inside a walkway", d["wk"].astype(bool), "walkway"), ("inside a carpark area", d["cp"].astype(bool), "carpark"),
             ("inside a map intersection (road_segment is_intersection)", d["inter"].astype(bool), "inter")]
    L += ["", "==== 2. where the predicted cells lie relative to the other GT layers of the same frame (shares of the cells of each group; 5-30 m, fov, image) ====",
          f"{'':64s} {'TRUE':>7s} {'FALSE':>7s} {'[95% CI]':>15s} {'near':>7s} {'far':>7s} | {'base: all fov cells':>19s} {'GT lane cells':>13s}",
          f"{'n cells':64s} {int(TRUE.sum()):7d} {int(FALSE.sum()):7d} {'':15s} {int(NEAR.sum()):7d} {int(FAR.sum()):7d} | {base['n']:19d} {gtl['n']:13d}"]
    R["layers"] = {}
    allrows = [(t, F[k], k) for t, k in rows] + other
    for t, arr, k in allrows:
        if k == "inter":
            bs, gs = base["inter"] / base["n"], gtl["inter"] / gtl["n"]
        else:
            bs, gs = bshare[k], gshare[k]
        rf = rate(arr & FALSE, FALSE)
        L.append(f"{t:64s} {arr[TRUE].mean():7.3f} {rf[0]:7.3f} {'[%.3f, %.3f]' % rf[1]:>15s} {arr[NEAR].mean():7.3f} {arr[FAR].mean():7.3f} | {bs:19.3f} {gs:13.3f}")
        R["layers"][k] = dict(true=float(arr[TRUE].mean()), false=rf[0], false_ci=rf[1], near=float(arr[NEAR].mean()), far=float(arr[FAR].mean()), base=float(bs), gt_lane=float(gs))
    un = F["unexpl"]
    rf = rate(un & FALSE, FALSE)
    L.append(f"{'NOT near (<= 1.0 m) any other labelled line / centreline, not in a box, not occluded':64s} {un[TRUE].mean():7.3f} {rf[0]:7.3f} {'[%.3f, %.3f]' % rf[1]:>15s} {un[NEAR].mean():7.3f} {un[FAR].mean():7.3f} | {bshare['unexpl']:19.3f} {gshare['unexpl']:13.3f}")
    R["layers"]["unexpl"] = dict(true=float(un[TRUE].mean()), false=rf[0], false_ci=rf[1], near=float(un[NEAR].mean()), far=float(un[FAR].mean()), base=float(bshare["unexpl"]), gt_lane=float(gshare["unexpl"]))
    dr_ = d["dr"].astype(bool)
    L += ["", "FALSE cells by range (shares): " + "; ".join(f"{lo:.0f}-{hi:.0f} m (n {int((FALSE & (x >= lo) & (x < hi)).sum())}): road_div<=1.0 m {F['rd10'][FALSE & (x >= lo) & (x < hi)].mean():.2f}, ped_cross<=1.0 m {F['pc10'][FALSE & (x >= lo) & (x < hi)].mean():.2f}, "
                                                          f"walk edge<=1.0 m {F['wb10'][FALSE & (x >= lo) & (x < hi)].mean():.2f}, centreline<=1.0 m {F['cl10'][FALSE & (x >= lo) & (x < hi)].mean():.2f}, in box {box[FALSE & (x >= lo) & (x < hi)].mean():.2f}, "
                                                          f"occluded {occ[FALSE & (x >= lo) & (x < hi)].mean():.2f}, outside drivable {(~dr_)[FALSE & (x >= lo) & (x < hi)].mean():.2f}, unexplained {un[FALSE & (x >= lo) & (x < hi)].mean():.2f}" for lo, hi in RANGES[:3])]
    # ---- 3. junction
    inter = d["inter"].astype(bool)
    L += ["", "==== 3. junction (nuScenes map road_segment polygons with is_intersection, rasterised into the ego grid) ====",
          f"share of predicted cells inside an intersection: TRUE {inter[TRUE].mean():.3f}, FALSE {inter[FALSE].mean():.3f} (near {inter[NEAR].mean():.3f}, far {inter[FAR].mean():.3f}); share of the GT lane_divider cells (fov, 5-30 m, image) inside: {gtl['inter'] / gtl['n']:.3f}; of all fov cells: {base['inter'] / base['n']:.3f}"]
    for nm, m in (("inside intersection", inter), ("outside intersection", ~inter)):
        r = rate(TRUE & m, m)
        L.append(f"precision {nm:21s}: {fmt(r)}  (n {r[2]} cells)")
        R.setdefault("junction", {})[nm] = dict(precision=r[0], ci=r[1], n=r[2])
    rt, rf = rate(inter & TRUE, TRUE), rate(inter & FALSE, FALSE)
    dd = diff(rf, rt)
    L.append(f"share inside: TRUE {fmt(rt)}, FALSE {fmt(rf)}, FALSE - TRUE {dd[0]:+.3f} [{dd[1][0]:+.3f}, {dd[1][1]:+.3f}]")
    R["junction"]["share_inside"] = dict(true=rt[0], false=rf[0], gt_lane=gtl["inter"] / gtl["n"], base=base["inter"] / base["n"])
    # ---- 3b. the same stripe and neighbour numbers split by intersection
    vm = clear & okr
    L += ["", "---- 3b. stripe rate and neighbours of the cells split by intersection (clear axis, 5-30 m) ----",
          f"{'where':8s} {'group':11s} {'n':>6s} {'stripe +-0.5 [95% CI]':>24s} | {'control':>7s} | rd<=1m  pc<=1m  stop<=1m  walk-edge<=1m  centreline<=1m  box  occluded  unexplained   (shares of all cells of the row's group, no clear/okr filter)"]
    R["by_intersection"] = {}
    for wname, wm in (("inside", inter), ("outside", ~inter)):
        for gname, gm in (("TRUE", TRUE), ("FALSE", FALSE), ("FALSE-near", NEAR), ("FALSE-far", FAR)):
            m = vm & wm & gm
            r5 = rate(d["hit05"] & m, m)
            cok, ch = d["c_ok"].astype(bool), d["c_hit05"].astype(bool)
            nc, dc = (ch & m[None]).sum(), (cok & m[None]).sum()
            a = wm & gm
            if a.sum() == 0:
                continue
            sh = [F[k][a].mean() for k in ("rd10", "pc10", "sl10", "wb10", "cl10")] + [box[a].mean(), occ[a].mean(), F["unexpl"][a].mean()]
            L.append(f"{wname:8s} {gname:11s} {r5[2]:6d} {fmt(r5):>24s} | {nc / max(dc, 1):7.3f} | " + "  ".join(f"{v:6.3f}" for v in sh) + f"   (n all {int(a.sum())})")
            R["by_intersection"][f"{wname}/{gname}"] = dict(n=r5[2], hit05=r5[0], ci=r5[1], control=float(nc / max(dc, 1)), shares=[float(v) for v in sh], n_all=int(a.sum()))
    open(os.path.join(HERE, "precision_paint_result.txt"), "w").write("\n".join(L) + "\n")
    json.dump(R, open(os.path.join(HERE, "precision_paint_result.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("\n".join(L))


if __name__ == "__main__":
    main()
