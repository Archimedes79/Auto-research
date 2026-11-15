"""scratch (analysis, no verdict): what does the lane_divider IoU become if the map's intersection polygons are left out? The 246 frames and stored dt-0 probability maps of the recall test
(rp_out), the metric's own counting replicated by hand with the same functions (evalsuite.comparison.maps.dilate / tolerant_counts_many, as evaluation/metrics.SegMeter._count: P = prob > 0.5,
G = GT > 0.5, mask = the GT camera (fov) mask, tolerances 0 / 0.5 / 1.0 m = disc of 0 / 1 / 2 cells, IoU = tp / (tp + p_miss + g_miss) with tp = (p_hit + g_hit) / 2).
Variants (the intersection raster = nuScenes map road_segment is_intersection, pp_precision.Maps; 'band' = the polygon dilated by 1 cell, i.e. also the cells touching it):
  base      the metric as is
  both      prediction, GT and evaluated mask lose the intersection cells (removed before the dilations)
  pred      predicted cells inside the intersection are removed from the prediction (GT and mask kept)
  fp_only   only the predicted cells inside the intersection that have no GT within tolerance are not counted as false (every other count as in base)
Paired scene bootstrap (2000 resamples, the same for every interval) of the difference to base. -> temp/overview/intersection_iou_result.txt / .json
python temp/overview/ii_intersection_iou.py"""
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pp_precision as PP  # noqa: E402
from evalsuite.comparison import maps as M  # noqa: E402

TOL = (0.0, 0.5, 1.0)
CELLS = tuple(t / 0.5 for t in TOL)
NB = 2000


def counts(P, G, cms, ignore_fp=None):
    """(tolerances, 4) counts [p_hit, p_miss, g_hit, g_miss] over the cells of cms; ignore_fp: cells whose predicted-false cells are not counted"""
    nears = [(M.dilate(G, r), M.dilate(P, r)) for r in CELLS]
    c = M.tolerant_counts_many(P, G, nears, [cms] if ignore_fp is None else [cms, cms & ~ignore_fp])
    if ignore_fp is None:
        return c[:, 0, :].astype(float)
    out = c[:, 0, :].astype(float)
    out[:, 1] = c[:, 1, 1]
    return out


def iou(c):
    tp = (c[..., 0] + c[..., 2]) / 2
    return tp / (tp + c[..., 1] + c[..., 3])


def main():
    frames = json.load(open(os.path.join(HERE, "rp_frames.json")))
    done = [f for f in frames if os.path.exists(os.path.join(HERE, "rp_out", f"{f['i']}.npz"))]
    scene_ids = {s: k for k, s in enumerate(sorted({f["scene"] for f in done}))}
    S = len(scene_ids)
    maps = PP.Maps()
    names = [f"{v}{b}" for b in ("", "+band") for v in ("both", "pred", "fp_only")]
    C = {n: np.zeros((len(done), 3, 4)) for n in ["base"] + names}
    stat = np.zeros((len(done), 8))        # fov cells, fov in I, fov in I+band, GT cells, GT in I, GT in I+band, pred cells, pred in I (pred in I+band in the next two columns below)
    stat2 = np.zeros((len(done), 2))
    stored_ok, max_diff = True, 0.0
    sc = np.zeros(len(done), int)
    for k, f in enumerate(done):
        i = f["i"]
        g = np.load(os.path.join(PP.ROOT, "data", "cache", "gt", f["file"]), allow_pickle=True)["gt"].item()["cur"]
        meta = g["meta"]
        prob = np.load(os.path.join(HERE, "rp_out", f"{i}.npz"))["lane_divider"].astype(np.float32)
        Pm = prob > 0.5
        Gl = np.asarray(g["lane_divider"], np.float32) > 0.5
        cam = np.asarray(g["camera"], np.float32) > 0
        inter = maps.raster(meta["scene"], meta["T_e2g"])
        band = M.dilate(inter, 1)
        sc[k] = scene_ids[f["scene"]]
        C["base"][k] = counts(Pm, Gl, cam)
        for b, I in (("", inter), ("+band", band)):
            C["both" + b][k] = counts(Pm & ~I, Gl & ~I, cam & ~I)
            C["pred" + b][k] = counts(Pm & ~I, Gl, cam)
            C["fp_only" + b][k] = counts(Pm, Gl, cam, ignore_fp=I)
        stat[k] = [cam.sum(), (cam & inter).sum(), (cam & band).sum(), (Gl & cam).sum(), (Gl & cam & inter).sum(), (Gl & cam & band).sum(), (Pm & cam).sum(), (Pm & cam & inter).sum()]
        stat2[k] = [(Pm & cam & band).sum(), 0]
        # the stored per-frame counts of the real metric (rp_out/<i>.counts.pkl)
        st = pickle.load(open(os.path.join(HERE, "rp_out", f"{i}.counts.pkl"), "rb"))
        j = [q for q, key in enumerate(st["keys"]) if str(key) == "fov|+0.00|lane_divider|all|all"][0]
        d = np.abs(st["cnt"][j].astype(float) - C["base"][k]).max()
        max_diff = max(max_diff, d)
    rng = np.random.RandomState(0)
    Wt = rng.multinomial(S, np.ones(S) / S, size=NB).astype(float)

    def per_scene(c):
        a = np.zeros((S, 3, 4))
        np.add.at(a, sc, c)
        return a
    PS = {n: per_scene(c) for n, c in C.items()}

    def boot(n):
        return np.einsum("bs,stc->btc", Wt, PS[n])
    Bb = boot("base")
    out, L = {}, []
    L += ["==== the metric's lane_divider IoU on the 246 frames, with the map's intersection polygons left out (replicated by hand; same dilation / counting functions as the metric) ====",
          f"replication check: max |my base counts - the real metric's stored per-frame counts| over all frames, tolerances and the four counts = {max_diff:.0f}"]
    tot = {n: C[n].sum(0) for n in C}
    L.append("baseline (tolerance 0 / 0.5 / 1.0 m): " + " / ".join(f"{iou(tot['base'][t]):.4f}" for t in range(3)) + "   (the real metric on the same frames: 0.1970 / 0.4462 / 0.5613)")
    out["replication"] = dict(max_abs_count_diff=float(max_diff), base_iou=[float(iou(tot["base"][t])) for t in range(3)])
    s = stat.sum(0)
    L += ["", "share removed (whole fov mask 0-60 m; 'band' = the polygon plus the cells touching it):",
          f"  evaluated fov cells: {s[1] / s[0]:.3f} inside the polygons, {s[2] / s[0]:.3f} with the band   ({int(s[0])} cells)",
          f"  GT lane_divider cells (fov): {s[4] / s[3]:.3f} inside, {s[5] / s[3]:.3f} with the band   ({int(s[3])} cells)",
          f"  predicted lane_divider cells (fov, p > 0.5): {s[7] / s[6]:.3f} inside, {stat2[:, 0].sum() / s[6]:.3f} with the band   ({int(s[6])} cells)"]
    out["removed"] = dict(fov=[s[1] / s[0], s[2] / s[0]], gt=[s[4] / s[3], s[5] / s[3]], pred=[s[7] / s[6], stat2[:, 0].sum() / s[6]])
    L += ["", f"{'variant':16s} {'tol m':>6s} {'IoU':>8s} {'95% CI (own)':>17s} | {'minus base':>10s} {'paired 95% CI':>17s} | {'precision':>9s} {'recall':>7s}   (precision = p_hit / (p_hit + p_miss), recall = g_hit / (g_hit + g_miss), at the same tolerance)"]
    out["variants"] = {}
    for n in ["base"] + names:
        for t in range(3):
            v = iou(boot(n)[:, t])
            ci = np.percentile(v, [2.5, 97.5])
            dv = v - iou(Bb[:, t])
            dci = np.percentile(dv, [2.5, 97.5])
            c = tot[n][t]
            pr, rc = c[0] / (c[0] + c[1]), c[2] / (c[2] + c[3])
            L.append(f"{n:16s} {TOL[t]:6.1f} {iou(c):8.4f} {'[%.3f, %.3f]' % tuple(ci):>17s} | {iou(c) - iou(tot['base'][t]):+10.4f} {'[%+.3f, %+.3f]' % tuple(dci):>17s} | {pr:9.3f} {rc:7.3f}")
            out["variants"].setdefault(n, {})[f"tol{TOL[t]}"] = dict(iou=float(iou(c)), ci=[float(x) for x in ci], diff=float(iou(c) - iou(tot["base"][t])), diff_ci=[float(x) for x in dci],
                                                                    precision=float(pr), recall=float(rc), counts=[float(x) for x in c])
        L.append("")
    open(os.path.join(HERE, "intersection_iou_result.txt"), "w").write("\n".join(L) + "\n")
    json.dump(out, open(os.path.join(HERE, "intersection_iou_result.json"), "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
