"""lane_readout.py - scratch tool (no verdict): a CPU read-out of the lane_divider map of a BEV run, on the 246 val frames of the paint check (temp/overview/rp_frames.json,
GT signature 6a7aa7864a), with the metric's own counting replicated by hand (same functions as evaluation/metrics.SegMeter: evalsuite.comparison.maps.dilate,
P = prob > 0.5, G = GT > 0.5, mask = the GT camera (fov) mask, tolerance 0 / 0.5 / 1.0 m = disc of 0 / 1 / 2 cells, IoU = tp / (tp + p_miss + g_miss), tp = (p_hit + g_hit) / 2).

    lane_readout.py run <run folder name> [--ckpt <file>] [--max-frames N] [--no-infer]     e.g.  lane_readout.py run t20_base_s0
    lane_readout.py compare <runA> <runB> [--ckpt-a <file>] [--ckpt-b <file>]                 paired differences B - A with scene-bootstrap 95 % intervals (2000 resamples)

run:     finds / makes the cached dt-0 maps of runs/<name> (last ckpt_*.pt or --ckpt), writes temp/overview/readout_<name>__<ckpt>.txt / .json / .npz:
         1 lane_divider IoU @ 0 / 0.5 / 1 m, precision / recall at each tolerance (+ check of the hand count against the real metric's stored counts)
         2 the same with the map's intersection polygons (road_segment is_intersection) left out of prediction, GT and mask: 'both', and 'both+band' (polygon + 1 cell border)
         3 recall of the GT label points (5-30 m ahead, as the paint check) at p > 0.5, 0.5 m: all / stripe / no stripe / not occluded / by range; control = 1.75 m sideways
         4 threshold sweep p = 0.3 .. 0.9: IoU @0.5 m, precision, recall (all cells; outside intersections); the best p is IN-SAMPLE
         5 average precision and ROC-AUC of the lane_divider probability (the real metric's logit-binned histogram: evaluation/bev_curves), on the fov cells
compare:  the same rows as differences B - A, paired by frame (same frames, same scene resamples). Needs the 'run' of both first (it runs it if the .npz is missing).
Inference: CPU only (CUDA_VISIBLE_DEVICES empty, 2 threads; run it under `nice -n 15`, e.g. via rp_py.sh), through evaluation/bev_adapter.build (the run's own code snapshot, EMA weights,
memory EMPTY = the card's protocol state 'reset', all three dt as the evaluation does, dt 0 kept) with the nuScenes devkit tables trimmed to the 83 scenes (rp_nusc_subset.py: 4 GB
peak instead of 9.5). About 3 s per frame, 12 min per run. A run without a 1 m head / without distance field / without a code snapshot is handled: missing outputs are skipped
and said so. Cached maps: temp/overview/readout_maps/<name>__<ckpt>/<frame>.npz (float32: lane_divider, lane_divider_dist, road_divider, ..., centreline) + <frame>.counts.pkl.
Static GT-derived arrays (GT lane / camera / occluded / other layers, intersection raster): temp/overview/readout_static.npz, made on first use (needs the map JSONs, 2 min).
Frame set: rp_frames.json / rp_points.npz (rp_stage1_points.py; the stripe flags of the paint check). Everything here is scratch and read-only on the repository.
Sanity (t20_base_s0, last checkpoint): IoU 0.1970 / 0.4462 / 0.5613 on the subset; with the intersections left out 0.220 / 0.509 / 0.641 (band: 0.222 / 0.514 / 0.647)."""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""                      # never the GPU (a training of another chat runs on it)
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, ".."), os.path.join(ROOT, "..", "autoresearch")]
SIG = "0.5_4_6a7aa7864a"
RES, X0, Y0, NX, NY = 0.5, 0.0, -26.0, 120, 104
TOLS = (0.0, 0.5, 1.0)
CELLS = (0, 1, 2)
PS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
NB = 2000
MAPS_DIR = os.path.join(HERE, "readout_maps")
STATIC = os.path.join(HERE, "readout_static.npz")
FROOT = os.path.expanduser("~/data/nuscenes_full")
RANGES = ((5.0, 12.0), (12.0, 20.0), (20.0, 30.0), (5.0, 30.0))
XC = (X0 + (np.arange(NX) + 0.5) * RES)[:, None] * np.ones((1, NY))
YC = np.ones((NX, 1)) * (Y0 + (np.arange(NY) + 0.5) * RES)[None, :]


# ---------------------------------------------------------------------------------------------------------------- frame set, static arrays
def load_frames():
    p = os.path.join(HERE, "rp_frames.json")
    if not os.path.exists(p):
        raise SystemExit("rp_frames.json / rp_points.npz are missing: run  bash temp/overview/rp_py.sh rp_stage1_points.py  first (the frame set and stripe flags of the paint check)")
    return json.load(open(p))


def dilate(m, r):
    from evalsuite.comparison import maps as M
    return M.dilate(m, r) if r > 0 else np.asarray(m, bool).copy()


class Maps:
    """intersection polygons of the nuScenes map of a scene's location (nuscenes.map_expansion, read only; the four map JSONs are 8-16 MB)"""

    def __init__(self):
        scenes = {s["token"]: s for s in json.load(open(os.path.join(FROOT, "v1.0-trainval", "scene.json")))}
        logs = {s["token"]: s for s in json.load(open(os.path.join(FROOT, "v1.0-trainval", "log.json")))}
        self.scene_loc = {t: logs[s["log_token"]]["location"] for t, s in scenes.items()}
        self.polys = {}

    def intersections(self, loc):
        if loc not in self.polys:
            from nuscenes.map_expansion.map_api import NuScenesMap
            nm = NuScenesMap(dataroot=FROOT, map_name=loc)
            self.polys[loc] = [(pg.bounds, pg) for pg in (nm.extract_polygon(r["polygon_token"]) for r in nm.road_segment if r.get("is_intersection"))]
            del nm
        return self.polys[loc]

    def raster(self, scene_token, T_e2g):
        from PIL import Image, ImageDraw
        polys = self.intersections(self.scene_loc[scene_token])
        T = np.asarray(T_e2g, float)
        ex, ey = T[0, 3], T[1, 3]
        Tinv = np.linalg.inv(T)
        ss = 4
        img = Image.new("L", (NY * ss, NX * ss), 0)
        dr = ImageDraw.Draw(img)

        def px(coords):
            c = np.asarray(coords, float)[:, :2]
            e = np.c_[c, np.zeros(len(c)), np.ones(len(c))] @ Tinv.T
            return [(float((y - Y0) / RES * ss), float((x - X0) / RES * ss)) for x, y in zip(e[:, 0], e[:, 1])]
        for (x0, y0, x1, y1), pg in polys:
            if x1 < ex - 90 or x0 > ex + 90 or y1 < ey - 90 or y0 > ey + 90:
                continue
            geoms = [pg] if pg.geom_type == "Polygon" else list(pg.geoms)
            for g in geoms:
                dr.polygon(px(g.exterior.coords), fill=1)
            for g in geoms:
                for it in g.interiors:
                    dr.polygon(px(it.coords), fill=0)
        return np.asarray(img, bool)[ss // 2::ss, ss // 2::ss][:NX, :NY]


LAYERS_STATIC = ("lane", "cam", "occ", "inter", "road", "ped", "stop", "walk", "drive", "cl", "box", "carpark")


def get_static(frames):
    """GT-derived bool arrays (F, 120, 104) of every frame of the list + the lane label points (rp_points.npz) with their cell, occlusion and intersection flags"""
    if os.path.exists(STATIC):
        z = np.load(STATIC, allow_pickle=True)
        if len(z["lane"]) == len(frames):
            return {k: z[k] for k in z.files}
    print("making readout_static.npz (GT arrays + intersection raster of every frame; 2 min) ...", flush=True)
    maps = Maps()
    out = {k: np.zeros((len(frames), NX, NY), bool) for k in LAYERS_STATIC}
    for k, f in enumerate(frames):
        g = np.load(os.path.join(ROOT, "data", "cache", "gt", f["file"]), allow_pickle=True)["gt"].item()["cur"]
        lay = lambda n: np.asarray(g[n], np.float32) > 0.5            # noqa: E731
        out["lane"][k], out["road"][k], out["ped"][k], out["stop"][k] = lay("lane_divider"), lay("road_divider"), lay("ped_crossing"), lay("stop_line")
        out["walk"][k], out["drive"][k], out["cl"][k], out["carpark"][k] = lay("walkway"), lay("drivable_area"), lay("centreline"), lay("carpark_area")
        out["cam"][k] = np.asarray(g["camera"], np.float32) > 0
        out["occ"][k] = np.asarray(g["occluded"], np.float32) > 0
        out["box"][k] = np.asarray(g["instance"]) >= 0
        out["inter"][k] = maps.raster(g["meta"]["scene"], g["meta"]["T_e2g"])
    pts = np.load(os.path.join(HERE, "rp_points.npz"))
    fi = pts["lane_divider/fi"].astype(int)
    x, y = pts["lane_divider/x"], pts["lane_divider/y"]
    ci, cj = np.floor((x - X0) / RES).astype(int), np.floor((y - Y0) / RES).astype(int)
    ci, cj = np.clip(ci, 0, NX - 1), np.clip(cj, 0, NY - 1)
    out.update(pt_fi=fi, pt_x=x, pt_y=y, pt_tx=pts["lane_divider/tx"], pt_ty=pts["lane_divider/ty"], pt_ci=ci, pt_cj=cj, pt_hit05=pts["lane_divider/hit05"].astype(bool),
               pt_hit02=pts["lane_divider/hit02"].astype(bool), pt_occ=out["occ"][fi, ci, cj], pt_inter=out["inter"][fi, ci, cj])
    np.savez(STATIC, **out)
    return out


# ---------------------------------------------------------------------------------------------------------------- inference
def resolve_ckpt(run, ckpt):
    d = os.path.join(ROOT, "runs", run)
    if not os.path.isdir(d):
        raise SystemExit(f"no run folder {d}")
    if ckpt:
        c = ckpt if os.path.isabs(ckpt) else (os.path.join(d, ckpt) if os.path.exists(os.path.join(d, ckpt)) else os.path.join(ROOT, ckpt))
        if not os.path.exists(c):
            raise SystemExit(f"no checkpoint {ckpt}")
        return c
    ck = sorted(glob.glob(os.path.join(d, "ckpt_*.pt")))
    if not ck:
        raise SystemExit(f"no ckpt_*.pt in {d}")
    return ck[-1]


def tag_of(run, ckpt_path):
    return f"{run}__{os.path.splitext(os.path.basename(ckpt_path))[0]}"


def infer(run, ckpt_path, tag, frames, limit=0):
    """dt-0 maps of the frames that have none yet (CPU, memory empty); returns the maps folder"""
    out = os.path.join(MAPS_DIR, tag)
    os.makedirs(out, exist_ok=True)
    todo = [f for f in (frames[:limit] if limit else frames) if not (os.path.exists(os.path.join(out, f"{f['i']}.npz")) and os.path.exists(os.path.join(out, f"{f['i']}.counts.pkl")))]
    if not todo:
        return out
    import torch
    torch.set_num_threads(2)
    assert not torch.cuda.is_available(), "CUDA visible: refusing to touch the GPU"
    sys.path.insert(0, HERE)
    import rp_nusc_subset
    rp_nusc_subset.restrict({f["scene"] for f in frames})
    from evaluation import bev_adapter as A
    os.chdir(ROOT)
    orig = A.glob

    class _G:                                                   # make load_run take THE checkpoint asked for
        @staticmethod
        def glob(pat, *a, **k):
            return [ckpt_path] if pat.endswith("ckpt_*.pt") else orig.glob(pat, *a, **k)
    A.glob = _G
    t0 = time.time()
    try:
        source, subject, task = A.build(f"runs/{run}", device="cpu", split="val", workers=0)
    finally:
        A.glob = orig
    d = source.describe()
    assert d["gt_signature"] == SIG, f"GT signature {d['gt_signature']} is not {SIG}: refusing (the dataset could try to build GT)"
    assert subject.device == "cpu"
    print(f"built {tag} in {time.time() - t0:.0f} s: {subject.describe()}", flush=True)
    seg = [m for m in task.metrics() if m.name == "seg"][0]
    by_tok = {r.key: r for sc in source.scenes() for r in source.frames(sc)}
    n = 0
    for f in todo:
        if f["token"] not in by_tok:
            continue
        t1 = time.time()
        sample = source.load(by_tok[f["token"]])
        subject.reset()
        pred = subject.predict(sample)
        o0 = pred.outputs["dt"][0.0]
        arrs = {k: np.asarray(v, np.float32) for k, v in o0["prob"].items() if k in ("lane_divider", "road_divider", "centreline")}
        arrs.update({f"{k}_dist": np.asarray(v, np.float32) for k, v in (o0.get("dist") or {}).items() if k in ("lane_divider", "road_divider")})
        np.savez(os.path.join(out, f"{f['i']}.tmp.npz"), **arrs)
        os.replace(os.path.join(out, f"{f['i']}.tmp.npz"), os.path.join(out, f"{f['i']}.npz"))
        pickle.dump(seg.frame(sample, pred), open(os.path.join(out, f"{f['i']}.counts.pkl"), "wb"))
        n += 1
        if n % 20 == 0:
            print(f"  {n}/{len(todo)} frames, {time.time() - t1:.1f} s per frame", flush=True)
    print(f"inference done: {n} new frames in {out}", flush=True)
    return out


# ---------------------------------------------------------------------------------------------------------------- counting
def tol_counts(P, G, cm, nG, nP):
    """(3, 4) [p_hit, p_miss, g_hit, g_miss] over the cells of cm; nG / nP = the dilations of G / P for the three tolerances"""
    return np.array([[(P & nG[t] & cm).sum(), (P & ~nG[t] & cm).sum(), (G & nP[t] & cm).sum(), (G & ~nP[t] & cm).sum()] for t in range(3)], float)


def iou(c):
    tp = (c[..., 0] + c[..., 2]) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return tp / (tp + c[..., 1] + c[..., 3])


def prec(c):
    with np.errstate(divide="ignore", invalid="ignore"):
        return c[..., 0] / (c[..., 0] + c[..., 1])


def rec(c):
    with np.errstate(divide="ignore", invalid="ignore"):
        return c[..., 2] / (c[..., 2] + c[..., 3])


def frame_arrays(frames_done, st, maps_dir, layer="lane_divider"):
    """the per-frame arrays of the read-out of a run"""
    from evaluation.bev_curves import histogram
    F = len(frames_done)
    A = {"cnt_base": np.zeros((F, 3, 4)), "cnt_both": np.zeros((F, 3, 4)), "cnt_band": np.zeros((F, 3, 4)), "sweep": np.zeros((F, len(PS), 2, 3, 4)), "hist": np.zeros((F, 5, 400)),
         "stat": np.zeros((F, 8)), "real_diff": np.full(F, np.nan)}
    found, ctl = [], []
    pt_idx = []
    for k, f in enumerate(frames_done):
        i = f["i"]
        z = np.load(os.path.join(maps_dir, f"{i}.npz"))
        prob = z[layer].astype(np.float32)
        G, cam, I = st["lane"][i], st["cam"][i], st["inter"][i]
        B = dilate(I, 1)
        P = prob > 0.5
        nG = [dilate(G, r) for r in CELLS]
        nP = [dilate(P, r) for r in CELLS]
        A["cnt_base"][k] = tol_counts(P, G, cam, nG, nP)
        for name, J in (("cnt_both", I), ("cnt_band", B)):
            Pj, Gj = P & ~J, G & ~J
            A[name][k] = tol_counts(Pj, Gj, cam & ~J, [dilate(Gj, r) for r in CELLS], [dilate(Pj, r) for r in CELLS])
        for ip, p in enumerate(PS):
            Pp = prob > p
            A["sweep"][k, ip, 0] = tol_counts(Pp, G, cam, nG, [dilate(Pp, r) for r in CELLS])
            Pj, Gj = Pp & ~I, G & ~I
            A["sweep"][k, ip, 1] = tol_counts(Pj, Gj, cam & ~I, [dilate(Gj, r) for r in CELLS], [dilate(Pj, r) for r in CELLS])
        A["hist"][k] = histogram(prob[cam], G[cam])
        A["stat"][k] = [cam.sum(), (cam & I).sum(), (cam & B).sum(), (G & cam).sum(), (G & cam & I).sum(), (G & cam & B).sum(), (P & cam).sum(), (P & cam & I).sum()]
        sel = np.nonzero(st["pt_fi"] == i)[0]
        pt_idx.append(sel)
        found.append(nP[1][st["pt_ci"][sel], st["pt_cj"][sel]])
        # control: the label point moved 1.75 m sideways (both sides)
        x2 = [st["pt_x"][sel] + s * (-st["pt_ty"][sel]) for s in (1.75, -1.75)]
        y2 = [st["pt_y"][sel] + s * st["pt_tx"][sel] for s in (1.75, -1.75)]
        c2 = []
        for xx, yy in zip(x2, y2):
            ci, cj = np.floor((xx - X0) / RES).astype(int), np.floor((yy - Y0) / RES).astype(int)
            ok = (ci >= 0) & (ci < NX) & (cj >= 0) & (cj < NY)
            c2.append(np.where(ok, nP[1][np.clip(ci, 0, NX - 1), np.clip(cj, 0, NY - 1)], False) * 1 + np.where(ok, 0, -1))
        ctl.append(np.stack(c2))
        # the real metric's stored counts (check of the hand count)
        pk = os.path.join(maps_dir, f"{i}.counts.pkl")
        if os.path.exists(pk):
            s_ = pickle.load(open(pk, "rb"))
            j = [q for q, key in enumerate(s_["keys"]) if str(key) == "fov|+0.00|lane_divider|all|all"]
            if j:
                A["real_diff"][k] = np.abs(s_["cnt"][j[0]].astype(float) - A["cnt_base"][k]).max()
    A["pt_sel"] = np.concatenate(pt_idx) if pt_idx else np.zeros(0, int)
    A["found"] = np.concatenate(found) if found else np.zeros(0, bool)
    A["ctl"] = np.concatenate(ctl, axis=1) if ctl else np.zeros((2, 0), int)
    return A


# ---------------------------------------------------------------------------------------------------------------- bootstrap
class Boot:
    def __init__(self, S, nb=NB, seed=0):
        rng = np.random.RandomState(seed)
        self.S, self.W = S, rng.multinomial(S, np.ones(S) / S, size=nb).astype(float)

    def scenes(self, arr, sc):
        """per-scene sums of arr (F, ...) by the scene index sc (F,)"""
        out = np.zeros((self.S,) + arr.shape[1:])
        np.add.at(out, sc, arr)
        return out

    def resample(self, per_scene):
        return np.tensordot(self.W, per_scene, axes=(1, 0))


def ci(v):
    v = v[np.isfinite(v)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) > 100 else (float("nan"), float("nan"))


def curves_numbers(h):
    from evaluation.bev_curves import classification_numbers
    r = classification_numbers(h)
    return r["ap"], r["roc_auc"]


def point_groups(st, sel, hit="pt_hit05"):
    """boolean masks over the selected label points: range bins x (all, stripe, no stripe, not occluded, not occluded & stripe, not occluded & no stripe, outside intersection)"""
    x, h, occ, inter = st["pt_x"][sel], st[hit][sel], st["pt_occ"][sel], st["pt_inter"][sel]
    groups = {}
    for lo, hi in RANGES:
        r = (x >= lo) & (x < hi)
        for gn, m in (("all", r), ("stripe", r & h), ("no stripe", r & ~h), ("not occluded", r & ~occ), ("not occluded, stripe", r & ~occ & h), ("not occluded, no stripe", r & ~occ & ~h),
                      ("outside intersection", r & ~inter), ("outside intersection, stripe", r & ~inter & h), ("outside intersection, no stripe", r & ~inter & ~h)):
            groups[(f"{lo:.0f}-{hi:.0f}", gn)] = m
    return groups


# ---------------------------------------------------------------------------------------------------------------- report of one run
def run_report(tag, frames, st, A, frames_done, maps_dir):
    scene_ids = {s: k for k, s in enumerate(sorted({f["scene"] for f in frames}))}
    sc = np.array([scene_ids[f["scene"]] for f in frames_done])
    bt = Boot(len(scene_ids))
    L, R = [], {"tag": tag, "n_frames": len(frames_done)}
    L.append(f"==== lane_readout {tag}: lane_divider read-out on {len(frames_done)} val frames (cold memory, EMA weights, CPU), GT {SIG} ====")
    rd = A["real_diff"]
    if np.isfinite(rd).any():
        L.append(f"hand count vs the real metric's stored counts: max |difference| over frames, tolerances and the four counts = {np.nanmax(rd):.0f} cells")
        R["replication_max_count_diff"] = float(np.nanmax(rd))
    s = A["stat"].sum(0)
    L += ["", "removed by the intersection polygons (whole fov mask 0-60 m; band = polygon + 1 cell): "
          f"fov cells {s[1] / s[0]:.3f} / {s[2] / s[0]:.3f}; GT lane cells {s[4] / s[3]:.3f} / {s[5] / s[3]:.3f}; predicted cells {s[7] / s[6]:.3f}"]
    R["removed"] = dict(fov=[s[1] / s[0], s[2] / s[0]], gt=[s[4] / s[3], s[5] / s[3]], pred=s[7] / s[6])
    L += ["", f"-- 1/2  IoU, precision, recall of the lane_divider map (p > 0.5; precision = p_hit / (p_hit + p_miss), recall = g_hit / (g_hit + g_miss) at the same tolerance); 95% CI over scenes (2000 resamples)",
          f"{'variant':22s} {'tol m':>5s} {'IoU':>7s} {'95% CI':>15s} {'precision':>9s} {'recall':>7s}"]
    R["iou"] = {}
    for key, name in (("cnt_base", "all cells"), ("cnt_both", "outside intersections"), ("cnt_band", "outside, + 1 cell band")):
        tot = A[key].sum(0)
        ps = bt.resample(bt.scenes(A[key], sc))
        for t in range(3):
            c_ = ci(iou(ps[:, t]))
            L.append(f"{name:22s} {TOLS[t]:5.1f} {iou(tot[t]):7.4f} {'[%.3f, %.3f]' % c_:>15s} {prec(tot[t]):9.3f} {rec(tot[t]):7.3f}")
            R["iou"].setdefault(key, {})[str(TOLS[t])] = dict(iou=float(iou(tot[t])), ci=c_, precision=float(prec(tot[t])), recall=float(rec(tot[t])))
    # recall of the label points
    sel = A["pt_sel"]
    groups = point_groups(st, sel)
    found, ctl = A["found"].astype(float), A["ctl"]
    pts_sc = np.array([scene_ids[frames[i]["scene"]] for i in st["pt_fi"][sel]]) if len(sel) else np.zeros(0, int)
    L += ["", "-- 3  recall of the GT lane_divider label points (5-30 m ahead, inside the image; p > 0.5, 0.5 m); stripe = contrast >= 0.25 within +-0.5 m (paint check); control = the same points moved 1.75 m sideways",
          f"{'range':8s} {'group':34s} {'n':>6s} {'recall':>7s} {'95% CI':>15s} | {'control':>7s}"]
    R["recall"] = {}
    for (rg, gn), m in groups.items():
        if m.sum() == 0:
            continue
        num, den = np.bincount(pts_sc, found * m, bt.S), np.bincount(pts_sc, m.astype(float), bt.S)
        r_, c_ = num.sum() / den.sum(), ci((bt.W @ num) / (bt.W @ den))
        okc = ctl >= 0
        cc = ((ctl == 1) & m[None]).sum() / max((okc & m[None]).sum(), 1)
        L.append(f"{rg:8s} {gn:34s} {int(den.sum()):6d} {r_:7.3f} {'[%.3f, %.3f]' % c_:>15s} | {cc:7.3f}")
        R["recall"][f"{rg}/{gn}"] = dict(n=int(den.sum()), recall=float(r_), ci=c_, control=float(cc))
    # threshold sweep
    L += ["", "-- 4  threshold sweep (tolerance 0.5 m; the best p of a column is chosen ON THESE SAME FRAMES = in-sample)",
          f"{'p':>5s} | {'IoU all':>8s} {'prec':>6s} {'rec':>6s} | {'IoU outside':>11s} {'prec':>6s} {'rec':>6s}"]
    R["sweep"] = {}
    sw = A["sweep"].sum(0)
    best = [int(np.argmax([iou(sw[ip, v, 1]) for ip in range(len(PS))])) for v in (0, 1)]
    for ip, p in enumerate(PS):
        a, b = sw[ip, 0, 1], sw[ip, 1, 1]
        L.append(f"{p:5.2f} | {iou(a):8.4f} {prec(a):6.3f} {rec(a):6.3f} | {iou(b):11.4f} {prec(b):6.3f} {rec(b):6.3f}" + ("   <- best (all cells, in-sample)" if ip == best[0] else "") + ("   <- best (outside, in-sample)" if ip == best[1] else ""))
        R["sweep"][f"{p:.2f}"] = dict(iou_all=float(iou(a)), iou_outside=float(iou(b)), prec_all=float(prec(a)), rec_all=float(rec(a)), prec_outside=float(prec(b)), rec_outside=float(rec(b)))
    # AP / AUC
    hs = bt.scenes(A["hist"], sc)
    ap, auc = curves_numbers(hs.sum(0))
    Hb = bt.resample(hs)
    bp = np.array([curves_numbers(h) for h in Hb])
    L += ["", f"-- 5  ranking of the probability on the fov cells (the real metric's logit-binned histograms): AP {ap:.4f} {'[%.3f, %.3f]' % ci(bp[:, 0])}, ROC-AUC {auc:.4f} {'[%.3f, %.3f]' % ci(bp[:, 1])}   (the card, 1000 frames: AP 0.2138 / AUC 0.8461 for t20_base_s0)"]
    R["curves"] = dict(ap=ap, ap_ci=ci(bp[:, 0]), roc_auc=auc, auc_ci=ci(bp[:, 1]))
    txt = "\n".join(L) + "\n"
    open(os.path.join(HERE, f"readout_{tag}.txt"), "w").write(txt)
    json.dump(R, open(os.path.join(HERE, f"readout_{tag}.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    np.savez(os.path.join(HERE, f"readout_{tag}.npz"), frames=np.array([f["i"] for f in frames_done]), **{k: v for k, v in A.items()})
    return txt


def do_run(run, ckpt, max_frames=0, no_infer=False):
    frames = load_frames()
    cp = resolve_ckpt(run, ckpt)
    tag = tag_of(run, cp)
    maps_dir = os.path.join(MAPS_DIR, tag)
    if not no_infer:
        infer(run, cp, tag, frames, max_frames)
    done = [f for f in (frames[:max_frames] if max_frames else frames) if os.path.exists(os.path.join(maps_dir, f"{f['i']}.npz"))]
    if not done:
        raise SystemExit(f"no cached maps in {maps_dir}")
    st = get_static(frames)
    for f in done[:1]:
        z = np.load(os.path.join(maps_dir, f"{f['i']}.npz"))
        if "lane_divider" not in z.files:
            raise SystemExit("this run has no lane_divider output")
        print("outputs of the run:", z.files, "(no distance field: the zero-crossing decoding is skipped)" if "lane_divider_dist" not in z.files else "", flush=True)
    A = frame_arrays(done, st, maps_dir)
    txt = run_report(tag, frames, st, A, done, maps_dir)
    print(txt)
    return tag


# ---------------------------------------------------------------------------------------------------------------- compare
def do_compare(ra, rb, ca=None, cb=None):
    frames = load_frames()
    tags = []
    for r, c in ((ra, ca), (rb, cb)):
        tag = tag_of(r, resolve_ckpt(r, c))
        if not os.path.exists(os.path.join(HERE, f"readout_{tag}.npz")):
            do_run(r, c)
        tags.append(tag)
    za, zb = (np.load(os.path.join(HERE, f"readout_{t}.npz")) for t in tags)
    common = sorted(set(za["frames"].tolist()) & set(zb["frames"].tolist()))
    ia = {f: k for k, f in enumerate(za["frames"].tolist())}
    ib = {f: k for k, f in enumerate(zb["frames"].tolist())}
    ka, kb = np.array([ia[f] for f in common]), np.array([ib[f] for f in common])
    st = get_static(frames)
    scene_ids = {s: k for k, s in enumerate(sorted({f["scene"] for f in frames}))}
    sc = np.array([scene_ids[frames[f]["scene"]] for f in common])
    bt = Boot(len(scene_ids))
    L = [f"==== lane_readout compare: B = {tags[1]}  minus  A = {tags[0]}   ({len(common)} common frames, paired scene bootstrap, 2000 resamples, 95% intervals) ===="]
    R = {"A": tags[0], "B": tags[1], "n_frames": len(common), "rows": {}}

    def row(name, a_v, b_v, a_ps, b_ps, fn):
        """a_ps / b_ps: per-scene sums (S, ...), fn: statistic of the summed array"""
        d_boot = fn(bt.resample(b_ps)) - fn(bt.resample(a_ps))
        a, b = float(fn(a_ps.sum(0, keepdims=True))[0]), float(fn(b_ps.sum(0, keepdims=True))[0])
        c_ = ci(d_boot)
        L.append(f"{name:46s} A {a:8.4f}   B {b:8.4f}   B-A {b - a:+8.4f}  [{c_[0]:+.4f}, {c_[1]:+.4f}]" + ("  *" if np.isfinite(c_[0]) and (c_[0] > 0 or c_[1] < 0) else ""))
        R["rows"][name] = dict(A=a, B=b, diff=b - a, ci=c_)
    L.append("(* = the interval excludes 0)")
    L += ["", "-- IoU / precision / recall (p > 0.5)"]
    for key, nm in (("cnt_base", "all cells"), ("cnt_both", "outside intersections"), ("cnt_band", "outside + band")):
        pa, pb = bt.scenes(za[key][ka], sc), bt.scenes(zb[key][kb], sc)
        for t in range(3):
            row(f"IoU@{TOLS[t]} {nm}", None, None, pa[:, t], pb[:, t], iou)
        row(f"precision@0.5 {nm}", None, None, pa[:, 1], pb[:, 1], prec)
        row(f"recall@0.5 {nm}", None, None, pa[:, 1], pb[:, 1], rec)
    L += ["", "-- threshold sweep, IoU@0.5 m (the winner of each side at its own best p is in-sample)"]
    for v, nm in ((0, "all cells"), (1, "outside intersections")):
        pa, pb = bt.scenes(za["sweep"][ka][:, :, v, 1], sc), bt.scenes(zb["sweep"][kb][:, :, v, 1], sc)
        for ip, p in enumerate(PS):
            row(f"IoU@0.5 {nm} p>{p:.2f}", None, None, pa[:, ip], pb[:, ip], iou)
    L += ["", "-- recall of the GT label points (p > 0.5, 0.5 m)"]
    pa_ids, pb_ids = za["pt_sel"], zb["pt_sel"]
    # points of the common frames in both runs (the selection is by frame, in the same order)
    fa, fb = za["found"].astype(float), zb["found"].astype(float)
    inA = np.isin(st["pt_fi"][pa_ids], common)
    inB = np.isin(st["pt_fi"][pb_ids], common)
    sel = pa_ids[inA]
    assert np.array_equal(sel, pb_ids[inB]), "the label points of the two runs differ"
    fa, fb = fa[inA], fb[inB]
    groups = point_groups(st, sel)
    pts_sc = np.array([scene_ids[frames[i]["scene"]] for i in st["pt_fi"][sel]])
    for (rg, gn), m in groups.items():
        if m.sum() < 30 or rg not in ("5-30", "5-12", "12-20", "20-30") or gn not in ("all", "stripe", "no stripe", "not occluded", "outside intersection"):
            continue
        den = np.bincount(pts_sc, m.astype(float), bt.S)
        na, nb = np.bincount(pts_sc, fa * m, bt.S), np.bincount(pts_sc, fb * m, bt.S)
        with np.errstate(divide="ignore", invalid="ignore"):
            d_boot = (bt.W @ nb) / (bt.W @ den) - (bt.W @ na) / (bt.W @ den)
        a_, b_ = na.sum() / den.sum(), nb.sum() / den.sum()
        c_ = ci(d_boot)
        nm = f"recall {rg} m, {gn} (n {int(den.sum())})"
        L.append(f"{nm:46s} A {a_:8.4f}   B {b_:8.4f}   B-A {b_ - a_:+8.4f}  [{c_[0]:+.4f}, {c_[1]:+.4f}]" + ("  *" if np.isfinite(c_[0]) and (c_[0] > 0 or c_[1] < 0) else ""))
        R["rows"][nm] = dict(A=float(a_), B=float(b_), diff=float(b_ - a_), ci=c_)
    L += ["", "-- ranking of the probability on the fov cells"]
    ha, hb = bt.scenes(za["hist"][ka], sc), bt.scenes(zb["hist"][kb], sc)
    Ha, Hb = bt.resample(ha), bt.resample(hb)
    pa_, pb_ = np.array([curves_numbers(h) for h in Ha]), np.array([curves_numbers(h) for h in Hb])
    for j, nm in enumerate(("AP", "ROC-AUC")):
        a, b = curves_numbers(ha.sum(0))[j], curves_numbers(hb.sum(0))[j]
        c_ = ci(pb_[:, j] - pa_[:, j])
        L.append(f"{nm:46s} A {a:8.4f}   B {b:8.4f}   B-A {b - a:+8.4f}  [{c_[0]:+.4f}, {c_[1]:+.4f}]" + ("  *" if np.isfinite(c_[0]) and (c_[0] > 0 or c_[1] < 0) else ""))
        R["rows"][nm] = dict(A=float(a), B=float(b), diff=float(b - a), ci=c_)
    name = f"compare_{tags[0]}_vs_{tags[1]}"
    txt = "\n".join(L) + "\n"
    open(os.path.join(HERE, f"readout_{name}.txt"), "w").write(txt)
    json.dump(R, open(os.path.join(HERE, f"readout_{name}.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("run")
    r.add_argument("--ckpt", default=None)
    r.add_argument("--max-frames", type=int, default=0)
    r.add_argument("--no-infer", action="store_true", help="use cached maps only")
    c = sub.add_parser("compare")
    c.add_argument("run_a")
    c.add_argument("run_b")
    c.add_argument("--ckpt-a", default=None)
    c.add_argument("--ckpt-b", default=None)
    a = ap.parse_args()
    if a.cmd == "run":
        do_run(a.run, a.ckpt, a.max_frames, a.no_infer)
    else:
        do_compare(a.run_a, a.run_b, a.ckpt_a, a.ckpt_b)


if __name__ == "__main__":
    main()
