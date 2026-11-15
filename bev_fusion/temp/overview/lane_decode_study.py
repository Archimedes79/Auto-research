"""lane_decode_study.py - scratch (diagnostic of DECODING, not a re-scoring of the networks): what do the threshold and the thinning of the lane_divider map do to the metric's IoU?
Works on the cached dt-0 maps of lane_readout.py (temp/overview/readout_maps/<run>__<ckpt>/; run `lane_readout.py run <run>` first), the metric's counting replicated by hand
(lane_readout.tol_counts: same dilation / counts as evaluation/metrics.SegMeter), all cells and outside the map's intersection polygons ('both': removed from prediction, GT and mask).

    lane_decode_study.py run <run> [--ckpt f]          -> decode_<run>__<ckpt>.txt / .json / .npz
    lane_decode_study.py arms <runA> <runB> [--ckpt-a f --ckpt-b f]   paired B - A, scene-bootstrap 95 % intervals (2000 resamples)

Decoders (p = threshold; baseline = thr@0.5):
  thr     prob > p                                   dil1   the thresholded mask dilated by one cell (a control: fatter)
  skel    skimage skeletonize of the thresholded mask  nms    non-maximum suppression across the local line direction: orientation = the dominant gradient direction of the
          Gaussian-smoothed (1 cell) map from the structure tensor (smoothed 2 cells); a cell stays if prob > p and its smoothed value is >= both neighbours one cell away along that normal
  zc      zero-crossing of the predicted distance field (the metric's own rule, evaluation.metrics.line_cells: local minimum along x or y and <= 0.25 m); only for a run whose adapter output
          has a 'lane_divider' distance field (lane_readout saves it as lane_divider_dist); scored against the GT lane mask, and with the metric's own zc F1 (GT zero-crossing cells)
Threshold study: IoU @0.5 m over p = 0.05 .. 0.99; the best p chosen on the SAME frames is IN-SAMPLE; held-out = p chosen on one half of the scenes and scored on the other half (and swapped),
repeated with a random split inside every bootstrap resample (the split keeps a scene's copies on one side)."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lane_readout as LR  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

PG = tuple(round(x, 2) for x in list(np.arange(0.05, 0.96, 0.05)) + [0.97, 0.99])
DEC_P = (0.5, 0.7, 0.9)
DECODERS = ("thr", "dil1", "skel", "nms")
TOLS = LR.TOLS
NB = LR.NB


def nms_map(prob, p, s1=1.0, s2=2.0):
    ps = ndi.gaussian_filter(prob, s1, mode="nearest")
    gx, gy = np.gradient(ps)
    jxx, jyy, jxy = (ndi.gaussian_filter(a, s2, mode="nearest") for a in (gx * gx, gy * gy, gx * gy))
    th = 0.5 * np.arctan2(2 * jxy, jxx - jyy)                     # direction of the dominant gradient = the normal of the line
    nx, ny = np.cos(th), np.sin(th)
    ii, jj = np.meshgrid(np.arange(prob.shape[0]), np.arange(prob.shape[1]), indexing="ij")
    a = ndi.map_coordinates(ps, [ii + nx, jj + ny], order=1, mode="nearest")
    b = ndi.map_coordinates(ps, [ii - nx, jj - ny], order=1, mode="nearest")
    return (prob > p) & (ps >= a) & (ps >= b)


def skel_map(prob, p):
    from skimage.morphology import skeletonize
    return skeletonize(prob > p)


def decode(name, prob, p):
    if name == "thr":
        return prob > p
    if name == "dil1":
        return LR.dilate(prob > p, 1)
    if name == "skel":
        return skel_map(prob, p)
    if name == "nms":
        return nms_map(prob, p)
    raise ValueError(name)


def study_arrays(tag, frames, st, maps_dir):
    from evaluation.metrics import line_cells
    done = [f for f in frames if os.path.exists(os.path.join(maps_dir, f"{f['i']}.npz"))]
    F = len(done)
    A = {"thr": np.zeros((F, len(PG), 2, 3, 4))}
    for d in DECODERS:
        for p in DEC_P:
            A[f"{d}@{p}"] = np.zeros((F, 2, 3, 4))
    has_dist = "lane_divider_dist" in np.load(os.path.join(maps_dir, f"{done[0]['i']}.npz")).files
    if has_dist:
        A["zc"], A["zc_metric"] = np.zeros((F, 2, 3, 4)), np.zeros((F, 3, 4))
    for k, f in enumerate(done):
        i = f["i"]
        z = np.load(os.path.join(maps_dir, f"{i}.npz"))
        prob = z["lane_divider"].astype(np.float32)
        G, cam, I = st["lane"][i], st["cam"][i], st["inter"][i]
        V = (("all", G, cam, np.zeros_like(I)), ("outside", G & ~I, cam & ~I, I))          # (name, GT, mask, cells removed from the prediction)
        nG = [[LR.dilate(g, r) for r in LR.CELLS] for _, g, _, _ in V]

        def count(P):
            out = np.zeros((2, 3, 4))
            for v, (_, g, cm, rem) in enumerate(V):
                Pv = P & ~rem
                out[v] = LR.tol_counts(Pv, g, cm, nG[v], [LR.dilate(Pv, r) for r in LR.CELLS])
            return out
        for ip, p in enumerate(PG):
            A["thr"][k, ip] = count(prob > p)
        for p in DEC_P:
            for d in DECODERS:
                A[f"{d}@{p}"][k] = count(decode(d, prob, p))
        if has_dist:
            dist = z["lane_divider_dist"].astype(np.float64)
            Pz = line_cells(dist, LR.RES)
            A["zc"][k] = count(Pz)
            g = np.load(os.path.join(LR.ROOT, "data", "cache", "gt", f["file"]), allow_pickle=True)["gt"].item()["cur"]
            Gz = line_cells(np.asarray(g["lane_divider_dist"], np.float32).astype(np.float64), LR.RES)
            A["zc_metric"][k] = LR.tol_counts(Pz, Gz, cam, [LR.dilate(Gz, r) for r in LR.CELLS], [LR.dilate(Pz, r) for r in LR.CELLS])
    return done, A


def scene_index(frames, done):
    ids = {s: k for k, s in enumerate(sorted({f["scene"] for f in frames}))}
    return np.array([ids[f["scene"]] for f in done]), len(ids)


# ---------------------------------------------------------------------------------------------------------------- held-out threshold choice
def heldout(psA, w, rng, base_ip, psB=None):
    """one 2-fold round: p chosen on one half of the scenes (those with weight > 0, weights = resample multiplicities), scored on the other half, both ways; the same split for the second arm.
    ps: per-scene counts (S, nPG, 3, 4). Returns (IoU@0.5 of the held-out choice, of the fixed baseline p, chosen indices) for arm A [and for arm B]"""
    sc = np.nonzero(w > 0)[0]
    perm = rng.permutation(sc)
    halves = (perm[:len(perm) // 2], perm[len(perm) // 2:])
    outs = []
    for ps in (psA,) + ((psB,) if psB is not None else ()):
        tot, base, ch = np.zeros((3, 4)), np.zeros((3, 4)), []
        for tr, te in ((halves[0], halves[1]), (halves[1], halves[0])):
            ctr = np.tensordot(w[tr], ps[tr], axes=(0, 0))
            ip = int(np.nanargmax(LR.iou(ctr[:, 1])))
            ch.append(ip)
            tot += np.tensordot(w[te], ps[te][:, ip], axes=(0, 0))
            base += np.tensordot(w[te], ps[te][:, base_ip], axes=(0, 0))
        outs.append((LR.iou(tot[1]), LR.iou(base[1]), ch))
    return outs


# ---------------------------------------------------------------------------------------------------------------- one run
def do_run(run, ckpt=None, quiet=False):
    frames = LR.load_frames()
    cp = LR.resolve_ckpt(run, ckpt)
    tag = LR.tag_of(run, cp)
    maps_dir = os.path.join(LR.MAPS_DIR, tag)
    if not os.path.isdir(maps_dir):
        raise SystemExit(f"no cached maps for {tag}: run  lane_readout.py run {run}  first")
    st = LR.get_static(frames)
    npz = os.path.join(HERE, f"decode_{tag}.npz")
    if os.path.exists(npz):
        z = np.load(npz)
        A, done = {k: z[k] for k in z.files if k != "frames"}, [frames[i] for i in z["frames"]]
    else:
        done, A = study_arrays(tag, frames, st, maps_dir)
        np.savez(npz, frames=np.array([f["i"] for f in done]), **A)
    sc, S = scene_index(frames, done)
    bt = LR.Boot(S)
    L, R = [], {"tag": tag, "n_frames": len(done)}
    iou, prec, rec, ci = LR.iou, LR.prec, LR.rec, LR.ci
    L.append(f"==== lane_decode_study {tag}: threshold and thinning of the lane_divider map, {len(done)} frames (hand count of the metric; 'outside' = intersection polygons removed from prediction, GT and mask) ====")
    # (a) thresholds
    ps = bt.scenes(A["thr"], sc)                                                    # (S, nPG, 2, 3, 4)
    tot = A["thr"].sum(0)
    ip0 = PG.index(0.5)
    L += ["", "-- (a) threshold p, IoU @0.5 m (precision / recall at 0.5 m); 'd vs 0.5' = paired difference to p = 0.5 with its scene-bootstrap interval; the best p is IN-SAMPLE (chosen on these frames)",
          f"{'p':>5s} | {'IoU all':>8s} {'prec':>6s} {'rec':>6s} {'d vs 0.5 [95% CI]':>24s} | {'IoU outside':>11s} {'prec':>6s} {'rec':>6s} {'d vs 0.5 [95% CI]':>24s}"]
    R["thr"] = {}
    best = [int(np.argmax(iou(tot[:, v, 1]))) for v in (0, 1)]
    boots = [bt.resample(ps[:, :, v]) for v in (0, 1)]                              # (NB, nPG, 3, 4)
    for ip, p in enumerate(PG):
        cells = []
        for v in (0, 1):
            d = iou(boots[v][:, ip, 1]) - iou(boots[v][:, ip0, 1])
            c = tot[ip, v, 1]
            cells.append(f"{iou(c):8.4f} {prec(c):6.3f} {rec(c):6.3f} {iou(c) - iou(tot[ip0, v, 1]):+8.4f} [{ci(d)[0]:+.4f}, {ci(d)[1]:+.4f}]" if v == 0 else
                         f"{iou(c):11.4f} {prec(c):6.3f} {rec(c):6.3f} {iou(c) - iou(tot[ip0, v, 1]):+8.4f} [{ci(d)[0]:+.4f}, {ci(d)[1]:+.4f}]")
            R["thr"].setdefault(f"{p}", {})[("all", "outside")[v]] = dict(iou=float(iou(c)), prec=float(prec(c)), rec=float(rec(c)), diff_vs_05=float(iou(c) - iou(tot[ip0, v, 1])), ci=ci(d))
        L.append(f"{p:5.2f} | " + " | ".join(cells) + ("   <- best all (in-sample)" if ip == best[0] else "") + ("   <- best outside (in-sample)" if ip == best[1] else ""))
    # held-out
    L += ["", "-- (a2) p chosen on half of the scenes, scored on the other half (swapped); gain over the fixed p = 0.5 on the same frames; the interval repeats the whole choice inside every bootstrap resample"]
    R["heldout"] = {}
    for v, nm in ((0, "all cells"), (1, "outside intersections")):
        rng = np.random.RandomState(1)
        pts = [heldout(ps[:, :, v], np.ones(S), rng, ip0)[0] for _ in range(200)]
        g0 = np.array([a - b for a, b, _ in pts])
        rng = np.random.RandomState(2)
        gb, chosen = [], []
        for w in bt.W[:1000]:
            a, b, ch = heldout(ps[:, :, v], w, rng, ip0)[0]
            gb.append(a - b)
            chosen += ch
        cnt = np.bincount(chosen, minlength=len(PG))
        L.append(f"{nm:22s}: held-out gain over p = 0.5: mean {g0.mean():+.4f} over 200 random splits (split-to-split range {g0.min():+.4f} .. {g0.max():+.4f}); with the choice repeated in 1000 bootstrap resamples: "
                 f"{np.mean(gb):+.4f} [{ci(np.array(gb))[0]:+.4f}, {ci(np.array(gb))[1]:+.4f}]; in-sample best p = {PG[best[v]]} gains {iou(tot[best[v], v, 1]) - iou(tot[ip0, v, 1]):+.4f}; chosen p (share of folds): "
                 + ", ".join(f"{PG[j]}:{cnt[j] / cnt.sum():.2f}" for j in np.argsort(-cnt)[:4]))
        R["heldout"][nm] = dict(mean_gain=float(g0.mean()), split_range=[float(g0.min()), float(g0.max())], boot_gain=float(np.mean(gb)), boot_ci=ci(np.array(gb)), best_p_insample=PG[best[v]],
                                insample_gain=float(iou(tot[best[v], v, 1]) - iou(tot[ip0, v, 1])))
    # (b) thinning
    L += ["", "-- (b) thinning of the output before scoring: IoU / precision / recall at tolerance 0 / 0.5 / 1.0 m; 'd' = paired difference of the IoU to the baseline decoding thr@0.5 [95% CI over scenes]",
          f"{'decoder':9s} {'variant':8s} | " + " | ".join(f"tol {t:.1f}: IoU  prec   rec  d [95% CI]" for t in TOLS)]
    base_ps = bt.resample(bt.scenes(A["thr"][:, ip0], sc))                            # (NB, 2, 3, 4)
    names = [f"{d}@{p}" for p in DEC_P for d in DECODERS] + (["zc"] if "zc" in A else [])
    R["decoders"] = {}
    for nm in names:
        a_ps = bt.resample(bt.scenes(A[nm], sc))
        tot_ = A[nm].sum(0)
        for v, vn in ((0, "all"), (1, "outside")):
            cells = []
            for t in range(3):
                d = iou(a_ps[:, v, t]) - iou(base_ps[:, v, t])
                cells.append(f"{iou(tot_[v, t]):.4f} {prec(tot_[v, t]):.3f} {rec(tot_[v, t]):.3f} {iou(tot_[v, t]) - iou(A['thr'][:, ip0].sum(0)[v, t]):+.4f} [{ci(d)[0]:+.3f},{ci(d)[1]:+.3f}]")
                R["decoders"].setdefault(nm, {}).setdefault(vn, {})[str(TOLS[t])] = dict(iou=float(iou(tot_[v, t])), prec=float(prec(tot_[v, t])), rec=float(rec(tot_[v, t])),
                                                                                         diff=float(iou(tot_[v, t]) - iou(A['thr'][:, ip0].sum(0)[v, t])), ci=ci(d))
            L.append(f"{nm:9s} {vn:8s} | " + " | ".join(cells))
    cells_n = {nm: A[nm].sum(0)[0, 1] for nm in names}
    L += ["", "predicted cells kept relative to thr@0.5 (all cells, p_hit + p_miss at 0.5 m): " + ", ".join(f"{nm} {(c[0] + c[1]) / (A['thr'][:, ip0].sum(0)[0, 1][0] + A['thr'][:, ip0].sum(0)[0, 1][1]):.2f}" for nm, c in cells_n.items())]
    # (c) distance field
    if "zc" in A:
        zm = A["zc_metric"].sum(0)
        f1 = 2 * prec(zm[1]) * rec(zm[1]) / (prec(zm[1]) + rec(zm[1]))
        L += ["", f"-- (c) the adapter exposes the lane_divider distance field (prediction dict 'dist'); zero-crossing cells are the row 'zc' of (b). The metric's own zero-crossing F1@0.5 m (GT zero-crossing cells): "
              f"{f1:.4f} (precision {prec(zm[1]):.3f}, recall {rec(zm[1]):.3f})  [the card of the 1000 frames: 0.1584; the subset with the real metric earlier: 0.1605]"]
        R["zc_metric_f1"] = float(f1)
    else:
        L += ["", "-- (c) this run's adapter output has NO lane_divider distance field: the zero-crossing decoding is skipped"]
    txt = "\n".join(L) + "\n"
    open(os.path.join(HERE, f"decode_{tag}.txt"), "w").write(txt)
    json.dump(R, open(os.path.join(HERE, f"decode_{tag}.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    if not quiet:
        print(txt)
    return tag


# ---------------------------------------------------------------------------------------------------------------- two arms
def do_arms(ra, rb, ca=None, cb=None):
    frames = LR.load_frames()
    tags = []
    for r, c in ((ra, ca), (rb, cb)):
        tags.append(do_run(r, c, quiet=True))
    za, zb = (np.load(os.path.join(HERE, f"decode_{t}.npz")) for t in tags)
    common = sorted(set(za["frames"].tolist()) & set(zb["frames"].tolist()))
    ia = {f: k for k, f in enumerate(za["frames"].tolist())}
    ib = {f: k for k, f in enumerate(zb["frames"].tolist())}
    ka, kb = np.array([ia[f] for f in common]), np.array([ib[f] for f in common])
    done = [frames[i] for i in common]
    sc, S = scene_index(frames, done)
    bt = LR.Boot(S)
    iou, ci = LR.iou, LR.ci
    A = {k: za[k][ka] for k in za.files if k != "frames"}
    B = {k: zb[k][kb] for k in zb.files if k != "frames"}
    L = [f"==== lane_decode_study arms: B = {tags[1]}  minus  A = {tags[0]}  ({len(common)} common frames; paired scene bootstrap, 2000 resamples, 95% intervals; * = excludes 0) ===="]
    R = {"A": tags[0], "B": tags[1], "rows": {}}

    def line(name, a, b, d):
        c = ci(d)
        L.append(f"{name:58s} A {a:7.4f}  B {b:7.4f}  B-A {b - a:+7.4f}  [{c[0]:+.4f}, {c[1]:+.4f}]" + ("  *" if np.isfinite(c[0]) and (c[0] > 0 or c[1] < 0) else ""))
        R["rows"][name] = dict(A=float(a), B=float(b), diff=float(b - a), ci=c)
    ip0 = PG.index(0.5)
    psA, psB = bt.scenes(A["thr"], sc), bt.scenes(B["thr"], sc)                    # (S, nPG, 2, 3, 4)
    totA, totB = A["thr"].sum(0), B["thr"].sum(0)
    L += ["", "-- IoU @0.5 m at a fixed threshold p (same p for both arms)"]
    for v, vn in ((0, "all cells"), (1, "outside intersections")):
        bA, bB = bt.resample(psA[:, :, v]), bt.resample(psB[:, :, v])
        for ip, p in enumerate(PG):
            if p in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99):
                line(f"IoU@0.5 {vn} p>{p}", iou(totA[ip, v, 1]), iou(totB[ip, v, 1]), iou(bB[:, ip, 1]) - iou(bA[:, ip, 1]))
        # each arm at its OWN best p (in-sample: chosen on these frames, in every resample on the resample itself)
        iA, iB = iou(bA[:, :, 1]), iou(bB[:, :, 1])
        ba, bb = int(np.argmax(iou(totA[:, v, 1]))), int(np.argmax(iou(totB[:, v, 1])))
        line(f"IoU@0.5 {vn}, each arm at its own best p [IN-SAMPLE] (A {PG[ba]}, B {PG[bb]})", iou(totA[ba, v, 1]), iou(totB[bb, v, 1]), iB.max(1) - iA.max(1))
        # held-out choice (same split for both arms; repeated in every resample)
        rng = np.random.RandomState(3)
        da, db = [], []
        pa0, pb0, rng0 = [], [], np.random.RandomState(4)
        for w in bt.W[:1000]:
            oa, ob = heldout(psA[:, :, v], w, rng, ip0, psB[:, :, v])
            da.append(oa[0]), db.append(ob[0])
        da, db = np.array(da), np.array(db)
        outs = [heldout(psA[:, :, v], np.ones(S), rng0, ip0, psB[:, :, v]) for _ in range(200)]
        line(f"IoU@0.5 {vn}, p chosen on half the scenes, scored on the other [held-out]", np.mean([o[0][0] for o in outs]), np.mean([o[1][0] for o in outs]), db - da)
    L += ["", "-- thinning decoders (IoU; paired difference B - A of the same decoder), all cells | outside intersections, tolerance 0.5 m and 1.0 m"]
    names = [k for k in A if "@" in k] + (["zc"] if "zc" in A and "zc" in B else [])
    for nm in names:
        ta, tb = A[nm].sum(0), B[nm].sum(0)
        bA, bB = bt.resample(bt.scenes(A[nm], sc)), bt.resample(bt.scenes(B[nm], sc))
        for v, vn in ((0, "all"), (1, "outside")):
            for t in (1, 2):
                line(f"{nm} {vn} tol {TOLS[t]}", iou(ta[v, t]), iou(tb[v, t]), iou(bB[:, v, t]) - iou(bA[:, v, t]))
    txt = "\n".join(L) + "\n"
    name = f"decode_arms_{tags[0]}_vs_{tags[1]}"
    open(os.path.join(HERE, f"{name}.txt"), "w").write(txt)
    json.dump(R, open(os.path.join(HERE, f"{name}.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("run")
    r.add_argument("--ckpt", default=None)
    c = sub.add_parser("arms")
    c.add_argument("run_a")
    c.add_argument("run_b")
    c.add_argument("--ckpt-a", default=None)
    c.add_argument("--ckpt-b", default=None)
    a = ap.parse_args()
    if a.cmd == "run":
        do_run(a.run, a.ckpt)
    else:
        do_arms(a.run_a, a.run_b, a.ckpt_a, a.ckpt_b)


if __name__ == "__main__":
    main()
