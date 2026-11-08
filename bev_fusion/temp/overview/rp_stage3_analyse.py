"""scratch (analysis, no verdict) stage 3 of the recall-by-paint test: per label point (lane_divider, road_divider; 5-30 m ahead; the points and stripe flags of
rp_stage1_points.py = the paint check) is the point FOUND by the network's dt-0 probability map (rp_stage2_infer.py)?
  found(p, tol) : some cell with probability >= p within `tol` of the point's GT cell (tol 0 = the cell itself, 1 cell = 0.5 m, 2 cells = 1.0 m; the DISC of
                  evalsuite.comparison.maps.dilate that the metric's tolerant IoU uses)
  control       : the same with the label point moved 1.75 m sideways (+-, along the line normal; points that leave the grid are dropped), grouped by the ORIGINAL point's stripe flag
  groups        : stripe visible = contrast >= 0.25 within +-0.5 m (hit05 of the paint check; the +-0.2 m flag as a second table), no stripe = the rest
  bootstrap     : over scenes (frames of one scene are correlated), 2000 resamples, 95 % percentile interval
Also: the real metric's lane_divider / road_divider IoU on the frames that were run (BevSegMetric.reduce of the stored per-frame counts) against the card value.
Output: temp/overview/recall_paint_result.txt and .json
python temp/overview/rp_stage3_analyse.py"""
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, ".."), os.path.join(ROOT, "..", "autoresearch")]
from evalsuite.comparison import maps as M  # noqa: E402

RES, X0, Y0, NX, NY = 0.5, 0.0, -26.0, 120, 104
PS = (0.3, 0.5, 0.7)
TOLS = (0, 1, 2)                                     # cells: exact, 0.5 m, 1.0 m
RANGES = ((5.0, 12.0), (12.0, 20.0), (20.0, 30.0))
SHIFT = 1.75
LAYERS = ("lane_divider", "road_divider")
NB = 2000
rng = np.random.RandomState(0)


def cell(x, y):
    return np.floor((x - X0) / RES).astype(int), np.floor((y - Y0) / RES).astype(int)


def inside(i, j):
    return (i >= 0) & (i < NX) & (j >= 0) & (j < NY)


def disc1_max(a):
    from scipy.ndimage import maximum_filter
    fp = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)           # radius 1 cell = 0.5 m
    return maximum_filter(a, footprint=fp, mode="constant", cval=0.0)


def main():
    frames = json.load(open(os.path.join(HERE, "rp_frames.json")))
    pts = np.load(os.path.join(HERE, "rp_points.npz"))
    done = {f["i"] for f in frames if os.path.exists(os.path.join(HERE, "rp_out", f"{f['i']}.npz")) and os.path.exists(os.path.join(HERE, "rp_out", f"{f['i']}.counts.pkl"))}
    scene_ids = {s: k for k, s in enumerate(sorted({f["scene"] for f in frames}))}
    print(f"{len(done)} of {len(frames)} frames have an output")
    R = {"n_frames_asked": len(frames), "n_frames_run": len(done), "layers": {}}
    # ---- per point results
    per = {}
    for layer in LAYERS:
        fi = pts[f"{layer}/fi"].astype(int)
        keep = np.isin(fi, list(done))
        d = {k: pts[f"{layer}/{k}"][keep] for k in ("fi", "x", "y", "tx", "ty", "hit02", "hit05")}
        n = len(d["x"])
        d["scene"] = np.array([scene_ids[frames[i]["scene"]] for i in d["fi"].astype(int)])
        d["found"] = {(p, t): np.zeros(n, bool) for p in PS for t in TOLS}
        d["found_gt"] = {(p, t): np.zeros(n, bool) for p in (0.5,) for t in TOLS}               # '>' instead of '>=' (the metric's rule), p = 0.5
        d["ctl"] = {(p, t): np.zeros((2, n), bool) for p in PS for t in TOLS}
        d["ctl_ok"] = np.zeros((2, n), bool)
        d["pe"] = np.full(n, np.nan)                                                             # probability in the point's own cell
        d["pm"] = np.full(n, np.nan)                                                             # max probability within 0.5 m
        d["ctl_pe"] = np.full((2, n), np.nan)
        d["gt_pos"] = np.zeros(n, bool)                                                          # the point's cell is a GT cell of the layer (mapping check)
        d["gt_cam"] = np.zeros(n, bool)                                                          # inside the camera (fov) mask of the GT
        d["gt_occ"] = np.zeros(n, bool)                                                          # marked occluded by a box in the GT
        d["in_grid"] = np.zeros(n, bool)
        per[layer] = d
    idx_by_frame = {layer: {} for layer in LAYERS}
    for layer in LAYERS:
        fi = per[layer]["fi"].astype(int)
        for i in np.unique(fi):
            idx_by_frame[layer][i] = np.nonzero(fi == i)[0]
    frames_seen = sorted(done)
    cellrec = {layer: np.zeros(3) for layer in LAYERS}                      # GT cells in the camera mask at 5-30 m: hits at 0.5 m, count, hits at 1.0 m (p > 0.5, the metric's rule)
    xcell = (X0 + (np.arange(NX) + 0.5) * RES)[:, None] * np.ones((1, NY))
    for k, i in enumerate(frames_seen):
        f = frames[i]
        out = np.load(os.path.join(HERE, "rp_out", f"{i}.npz"))
        g = np.load(os.path.join(ROOT, "data", "cache", "gt", f["file"]), allow_pickle=True)["gt"].item()["cur"]
        for layer in LAYERS:
            Pm = out[layer].astype(np.float32) > 0.5
            Gm = (np.asarray(g[layer], np.float32) > 0.5) & (np.asarray(g["camera"], np.float32) > 0) & (xcell >= 5) & (xcell < 30)
            cellrec[layer] += [(M.dilate(Pm, 1) & Gm).sum(), Gm.sum(), (M.dilate(Pm, 2) & Gm).sum()]
        for layer in LAYERS:
            sel = idx_by_frame[layer].get(i)
            if sel is None:
                continue
            d = per[layer]
            prob = out[layer].astype(np.float32)
            ci, cj = cell(d["x"][sel], d["y"][sel])
            ok = inside(ci, cj)
            d["in_grid"][sel] = ok
            ci, cj = np.clip(ci, 0, NX - 1), np.clip(cj, 0, NY - 1)
            d["gt_pos"][sel] = ok & (np.asarray(g[layer], np.float32)[ci, cj] > 0.5)
            d["gt_cam"][sel] = ok & (np.asarray(g["camera"], np.float32)[ci, cj] > 0)
            d["gt_occ"][sel] = ok & (np.asarray(g["occluded"], np.float32)[ci, cj] > 0)
            d["pe"][sel] = prob[ci, cj]
            d["pm"][sel] = disc1_max(prob)[ci, cj]
            for p in PS:
                B = prob >= p
                for t in TOLS:
                    D = M.dilate(B, t) if t else B
                    d["found"][(p, t)][sel] = D[ci, cj] & ok
                    if p == 0.5:
                        Bg = prob > p
                        d["found_gt"][(p, t)][sel] = (M.dilate(Bg, t) if t else Bg)[ci, cj] & ok
            for s, sh in enumerate((SHIFT, -SHIFT)):
                x2 = d["x"][sel] + sh * (-d["ty"][sel])
                y2 = d["y"][sel] + sh * d["tx"][sel]
                ci2, cj2 = cell(x2, y2)
                ok2 = inside(ci2, cj2)
                d["ctl_ok"][s, sel] = ok2
                ci2, cj2 = np.clip(ci2, 0, NX - 1), np.clip(cj2, 0, NY - 1)
                d["ctl_pe"][s, sel] = prob[ci2, cj2]
                for p in PS:
                    B = prob >= p
                    for t in TOLS:
                        D = M.dilate(B, t) if t else B
                        d["ctl"][(p, t)][s, sel] = D[ci2, cj2] & ok2
        if (k + 1) % 50 == 0:
            print(f"  analysed {k + 1}/{len(frames_seen)}", flush=True)

    # ---- tables
    lines = []
    out = {}

    def binsel(d, lo, hi):
        return (d["x"] >= lo) & (d["x"] < hi)

    def boot(d, mask_a, mask_b, key, ctl=False):
        """recall of group a, of group b and a - b, and the scene-bootstrap 95 % intervals; ctl: the control (shifted) points"""
        S = len(scene_ids)
        if ctl:
            fa, fb = d["ctl"][key].astype(float), d["ctl"][key].astype(float)
            va = d["ctl_ok"]
            num_a = np.stack([np.bincount(d["scene"], (fa[s] * (va[s] & mask_a)), S) for s in range(2)]).sum(0)
            den_a = np.stack([np.bincount(d["scene"], (va[s] & mask_a).astype(float), S) for s in range(2)]).sum(0)
            num_b = np.stack([np.bincount(d["scene"], (fb[s] * (va[s] & mask_b)), S) for s in range(2)]).sum(0)
            den_b = np.stack([np.bincount(d["scene"], (va[s] & mask_b).astype(float), S) for s in range(2)]).sum(0)
        else:
            f = d["found"][key].astype(float)
            num_a, den_a = np.bincount(d["scene"], f * mask_a, S), np.bincount(d["scene"], mask_a.astype(float), S)
            num_b, den_b = np.bincount(d["scene"], f * mask_b, S), np.bincount(d["scene"], mask_b.astype(float), S)
        ra = num_a.sum() / den_a.sum() if den_a.sum() else np.nan
        rb = num_b.sum() / den_b.sum() if den_b.sum() else np.nan
        W = rng.multinomial(S, np.ones(S) / S, size=NB).astype(float)           # scene resample weights
        with np.errstate(divide="ignore", invalid="ignore"):
            A = (W @ num_a) / (W @ den_a)
            Bq = (W @ num_b) / (W @ den_b)
        ci = lambda v: tuple(np.nanpercentile(v, [2.5, 97.5])) if np.isfinite(v).sum() > 100 else (np.nan, np.nan)      # noqa: E731
        return dict(n_a=int(den_a.sum()), n_b=int(den_b.sum()), ra=float(ra), rb=float(rb), diff=float(ra - rb), ci_a=ci(A), ci_b=ci(Bq), ci_diff=ci(A - Bq))

    for layer in LAYERS:
        d = per[layer]
        n = len(d["x"])
        R["layers"][layer] = {"n_points": int(n), "n_scenes": int(len(np.unique(d["scene"]))), "n_frames_with_points": int(len(np.unique(d["fi"])))}
        lines += ["", f"==== {layer}: {n} label points of {len(np.unique(d['fi']))} frames / {len(np.unique(d['scene']))} scenes (5-30 m ahead, inside the image) ====",
                  f"mapping check: {d['gt_pos'].mean():.3f} of the label points lie in a GT cell of the layer; {d['gt_cam'].mean():.3f} in the camera (fov) mask; {d['gt_occ'].mean():.3f} marked occluded by a box in the GT; {d['in_grid'].mean():.3f} in the grid",
                  "stripe share (hit05 / hit02): " + ", ".join(f"{lo:.0f}-{hi:.0f} m {d['hit05'][binsel(d, lo, hi)].mean():.2f} / {d['hit02'][binsel(d, lo, hi)].mean():.2f} (n {int(binsel(d, lo, hi).sum())})" for lo, hi in RANGES)
                  + f", all {d['hit05'][(d['x'] >= 5) & (d['x'] < 30)].mean():.2f} / {d['hit02'][(d['x'] >= 5) & (d['x'] < 30)].mean():.2f}"]
        for flag, label in (("hit05", "stripe within +-0.5 m"), ("hit02", "stripe within +-0.2 m")):
            for p in PS:
                if flag == "hit02" and p != 0.5:
                    continue
                lines += ["", f"-- {layer}, p >= {p}, stripe = {label}; tolerance 0.5 m (found = a cell with p >= {p} within 0.5 m of the label cell); control = label moved {SHIFT} m sideways (same points, original flag)",
                          f"{'range':8s} {'group':10s} {'n':>6s} {'recall 0.5m':>11s} {'95% CI':>15s} | {'control':>7s} | {'exact cell':>10s} {'1.0 m':>7s} | {'median p':>8s} {'median max p<=0.5m':>18s}"]
                for lo, hi in RANGES + ((5.0, 30.0),):
                    bm = binsel(d, lo, hi)
                    ha = d[flag].astype(bool)
                    key = (p, 1)
                    b = boot(d, bm & ha, bm & ~ha, key)
                    bc = boot(d, bm & ha, bm & ~ha, key, ctl=True)
                    b_ex, b_1m = boot(d, bm & ha, bm & ~ha, (p, 0)), boot(d, bm & ha, bm & ~ha, (p, 2))
                    for grp, m, ra, ci_, rc, rex, r1m, nn in (("stripe", bm & ha, b["ra"], b["ci_a"], bc["ra"], b_ex["ra"], b_1m["ra"], b["n_a"]),
                                                          ("no stripe", bm & ~ha, b["rb"], b["ci_b"], bc["rb"], b_ex["rb"], b_1m["rb"], b["n_b"])):
                        lines.append(f"{lo:3.0f}-{hi:<4.0f} {grp:10s} {nn:6d} {ra:11.3f} {'[%.3f, %.3f]' % ci_:>15s} | {rc:7.3f} | {rex:10.3f} {r1m:7.3f} | {np.nanmedian(d['pe'][m]):8.3f} {np.nanmedian(d['pm'][m]):18.3f}")
                    lines.append(f"{'':8s} {'diff':10s} {'':6s} {b['diff']:+11.3f} {'[%+.3f, %+.3f]' % b['ci_diff']:>15s} | {bc['diff']:+7.3f} | {b_ex['diff']:+10.3f} {b_1m['diff']:+7.3f} |   (diff = stripe - no stripe; 95% CI over scenes; control diff CI {'[%+.3f, %+.3f]' % bc['ci_diff']})")
                    R["layers"][layer].setdefault(f"{flag}/p{p}", {})[f"{lo:.0f}-{hi:.0f}"] = dict(
                        n_stripe=b["n_a"], n_nostripe=b["n_b"], recall_tol05=dict(stripe=b["ra"], nostripe=b["rb"], diff=b["diff"], ci_stripe=b["ci_a"], ci_nostripe=b["ci_b"], ci_diff=b["ci_diff"]),
                        control_tol05=dict(stripe=bc["ra"], nostripe=bc["rb"], diff=bc["diff"], ci_diff=bc["ci_diff"]),
                        exact=dict(stripe=b_ex["ra"], nostripe=b_ex["rb"], diff=b_ex["diff"], ci_diff=b_ex["ci_diff"]),
                        tol10=dict(stripe=b_1m["ra"], nostripe=b_1m["rb"], diff=b_1m["diff"], ci_diff=b_1m["ci_diff"]),
                        median_p_exact=dict(stripe=float(np.nanmedian(d["pe"][bm & ha])), nostripe=float(np.nanmedian(d["pe"][bm & ~ha]))),
                        median_pmax05=dict(stripe=float(np.nanmedian(d["pm"][bm & ha])), nostripe=float(np.nanmedian(d["pm"][bm & ~ha]))))
        # all points together (no stripe split), and the strict '>' rule at p = 0.5
        allm = (d["x"] >= 5) & (d["x"] < 30)
        lines += ["", f"-- {layer}: recall of ALL label points 5-30 m (no split), tolerance 0.5 m: " + ", ".join(f"p>={p}: {d['found'][(p, 1)][allm].mean():.3f} (control {d['ctl'][(p, 1)][d['ctl_ok'] & allm[None]].mean():.3f})" for p in PS)
                  + f"; with the metric's strict p > 0.5 rule: {d['found_gt'][(0.5, 1)][allm].mean():.3f}; exact cell p>=0.5: {d['found'][(0.5, 0)][allm].mean():.3f}; 1.0 m: {d['found'][(0.5, 2)][allm].mean():.3f}"]
        # camera-fov / occlusion split of the main comparison at p = 0.5, tolerance 0.5
        lines += [f"-- {layer}, p >= 0.5, tol 0.5 m, 5-30 m: stripe x (GT marks the cell occluded by a box)"]
        for occ, ol in ((False, "not occluded"), (True, "occluded")):
            for st, sl in ((True, "stripe"), (False, "no stripe")):
                m = allm & (d["gt_occ"] == occ) & (d["hit05"].astype(bool) == st)
                lines.append(f"   {ol:13s} {sl:10s} n {int(m.sum()):6d}  recall {d['found'][(0.5, 1)][m].mean() if m.sum() else float('nan'):.3f}   control {d['ctl'][(0.5, 1)][d['ctl_ok'] & m[None]].mean() if m.sum() else float('nan'):.3f}")
        lines.append(f"-- {layer}: recall only for label points inside the GT camera (fov) mask: stripe {d['found'][(0.5, 1)][allm & d['gt_cam'] & d['hit05'].astype(bool)].mean():.3f} (n {int((allm & d['gt_cam'] & d['hit05'].astype(bool)).sum())}), "
                     f"no stripe {d['found'][(0.5, 1)][allm & d['gt_cam'] & ~d['hit05'].astype(bool)].mean():.3f} (n {int((allm & d['gt_cam'] & ~d['hit05'].astype(bool)).sum())})")
        # unique cells (the same cell hit by several points)
        cells = {}
        for q in np.nonzero(allm)[0]:
            ci, cj = cell(d["x"][q], d["y"][q])
            cells.setdefault((int(d["fi"][q]), int(ci), int(cj)), []).append(q)
        multi = sum(1 for v in cells.values() if len(v) > 1)
        lines.append(f"-- {layer}: {int(allm.sum())} points fall into {len(cells)} distinct (frame, cell) pairs ({multi} cells hold more than one point; points are every 0.5 m along the line, diagonal runs and duplicated lines share cells)")

        # ---- extra 1: points not hidden behind a box (a 'stripe' found on a vehicle's body is not paint), p >= 0.5, tol 0.5 m
        # ---- extra 2: dash-gap candidates: no stripe at the point, but a stripe point of the same line within 6 m along it (|along| <= 6 m, |lateral| <= 0.75 m)
        near = np.zeros(n, bool)
        for fi_, sel in idx_by_frame[layer].items():
            xs, ys, txs, tys, hs = d["x"][sel], d["y"][sel], d["tx"][sel], d["ty"][sel], d["hit05"][sel].astype(bool)
            dx, dy = xs[None, :] - xs[:, None], ys[None, :] - ys[:, None]                       # [i, j] = point j seen from point i
            along = dx * txs[:, None] + dy * tys[:, None]
            lat = np.abs(-dx * tys[:, None] + dy * txs[:, None])
            ok_ = (np.abs(along) <= 6.0) & (lat <= 0.75) & hs[None, :] & ~np.eye(len(sel), dtype=bool)
            near[sel] = ok_.any(1)
        st = d["hit05"].astype(bool)
        R["layers"][layer]["extra"] = {}
        for gname, base in (("all points", np.ones(n, bool)), ("not occluded", ~d["gt_occ"])):
            lines += ["", f"-- {layer}, p >= 0.5, tol 0.5 m, {gname}: stripe / no stripe split further by 'a stripe of the same line within 6 m' (dash-gap candidate) or not (isolated: no paint seen anywhere near)",
                      f"{'range':8s} {'group':32s} {'n':>6s} {'recall 0.5m':>11s} {'95% CI':>15s} | {'control':>7s}"]
            for lo, hi in RANGES + ((5.0, 30.0),):
                bm = binsel(d, lo, hi) & base
                gm = {"stripe": bm & st, "no stripe, stripe within 6 m": bm & ~st & near, "no stripe, none within 6 m": bm & ~st & ~near}
                for gn, m in gm.items():
                    b = boot(d, m, bm & ~m, (0.5, 1))
                    bc = boot(d, m, bm & ~m, (0.5, 1), ctl=True)
                    lines.append(f"{lo:3.0f}-{hi:<4.0f} {gn:32s} {b['n_a']:6d} {b['ra']:11.3f} {'[%.3f, %.3f]' % b['ci_a']:>15s} | {bc['ra']:7.3f}")
                    R["layers"][layer]["extra"][f"{gname}/{lo:.0f}-{hi:.0f}/{gn}"] = dict(n=b["n_a"], recall=b["ra"], ci=b["ci_a"], control=bc["ra"])
                b = boot(d, bm & st, bm & ~st, (0.5, 1))
                lines.append(f"{'':8s} {'stripe - no stripe (all no-stripe)':32s} {'':6s} {b['diff']:+11.3f} {'[%+.3f, %+.3f]' % b['ci_diff']:>15s} |")
                R["layers"][layer]["extra"][f"{gname}/{lo:.0f}-{hi:.0f}/diff"] = dict(diff=b["diff"], ci=b["ci_diff"])
                b = boot(d, bm & st, bm & ~st & ~near, (0.5, 1))
                lines.append(f"{'':8s} {'stripe - isolated no stripe':32s} {'':6s} {b['diff']:+11.3f} {'[%+.3f, %+.3f]' % b['ci_diff']:>15s} |")
                R["layers"][layer]["extra"][f"{gname}/{lo:.0f}-{hi:.0f}/diff_isolated"] = dict(diff=b["diff"], ci=b["ci_diff"])
        lines += ["", f"-- {layer}: recall of all label points per range (p >= 0.5, tol 0.5 m): " + ", ".join(f"{lo:.0f}-{hi:.0f} m {d['found'][(0.5, 1)][binsel(d, lo, hi)].mean():.3f}" for lo, hi in RANGES)
                  + f"; share of no-stripe points that have a stripe within 6 m along the line: {near[~st].mean():.2f}"]
        cr = cellrec[layer]
        lines.append(f"-- {layer}: cross-check with the metric's own cell logic: of the GT cells of the layer in the camera mask at 5-30 m, {cr[0] / cr[1]:.3f} have a predicted cell (p > 0.5) within 0.5 m ({int(cr[1])} cells); 1.0 m: {cr[2] / cr[1]:.3f}")

    # ---- IoU of the real metric on the frames that were run
    from evaluation.bev_task import BevSegMetric
    counts = [pickle.load(open(os.path.join(HERE, "rp_out", f"{i}.counts.pkl"), "rb")) for i in frames_seen]
    scene_of = [frames[i]["scene"] for i in frames_seen]
    seg = BevSegMetric(None, None)
    KEYS = ["fov/dt+0.00/lane_divider/all/all/iou@0", "fov/dt+0.00/lane_divider/all/all/iou@0.5", "fov/dt+0.00/lane_divider/all/all/iou@1",
            "fov/dt+0.00/lane_divider/all/0-20/iou@0.5", "fov/dt+0.00/lane_divider/all/20-40/iou@0.5", "fov/dt+0.00/lane_divider/all/40-60/iou@0.5",
            "fov/dt+0.00/road_divider/all/all/iou@0.5", "fov/dt+0.00/road_divider/all/all/iou@1", "fov/dt+0.00/lane_divider:zc/all/all/f1@0.5"]
    val = seg.reduce(counts, keys=KEYS)
    card = json.load(open(os.path.join(ROOT, "runs", "t20_base_s0", "eval", "card.json")))["metrics"]
    sc_idx = {}
    for k, s in enumerate(scene_of):
        sc_idx.setdefault(s, []).append(k)
    names = sorted(sc_idx)
    bootv = {k: [] for k in KEYS}
    for _ in range(200):
        pick = rng.randint(0, len(names), len(names))
        cc = [counts[q] for j in pick for q in sc_idx[names[j]]]
        v = seg.reduce(cc, keys=KEYS)
        for k in KEYS:
            bootv[k].append(v[k])
    lines += ["", f"==== the real metric on the {len(frames_seen)} frames that were run (cold memory, EMA weights, CPU) against the card (1000 val frames, protocol state reset) ====",
              f"{'metric':52s} {'subset':>8s} {'95% CI (scenes)':>17s} {'card':>8s}"]
    R["iou"] = {}
    for k in KEYS:
        ck = "seg_fov/seg/" + k.split("/", 1)[1]
        lo_, hi_ = np.nanpercentile(bootv[k], [2.5, 97.5])
        lines.append(f"{k:52s} {val[k]:8.4f} {'[%.3f, %.3f]' % (lo_, hi_):>17s} {card.get(ck, float('nan')):8.4f}")
        R["iou"][k] = dict(subset=float(val[k]), ci=(float(lo_), float(hi_)), card=card.get(ck))
    # tolerant precision / recall of the metric itself at 0.5 m (from the counts)
    acc = seg._accumulate(counts, {"fov|+0.00|lane_divider|all|all", "fov|+0.00|road_divider|all|all"})
    for nm in ("lane_divider", "road_divider"):
        cnt = acc[f"fov|+0.00|{nm}|all|all"][0]                       # (tolerances, 4): p_hit, p_miss, g_hit, g_miss
        for ti, tl in ((1, 0.5), (2, 1.0)):
            ph, pm_, gh, gm = cnt[ti]
            lines.append(f"{nm}: tolerance {tl} m, cells of the whole fov mask: precision (predicted cells within {tl} m of a GT cell) {ph / (ph + pm_):.3f}, recall (GT cells with a predicted cell within {tl} m) {gh / (gh + gm):.3f}")
            R["iou"][f"{nm}/pr@{tl}"] = dict(precision=float(ph / (ph + pm_)), recall=float(gh / (gh + gm)))

    txt = "\n".join(lines) + "\n"
    open(os.path.join(HERE, "recall_paint_result.txt"), "w").write(txt)
    json.dump(R, open(os.path.join(HERE, "recall_paint_result.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(txt)


if __name__ == "__main__":
    main()
