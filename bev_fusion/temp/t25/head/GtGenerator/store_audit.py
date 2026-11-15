"""Audit of a stored lane GT: does the store hold every keyframe of the nuScenes version, does every file load and hold what the version promises. Used for every store before it is released
(l4, l5: 34,149 files, 4 - 8 min on one core); `python -m GtGenerator audit-lane SYSTEM`.

    r = store_audit.audit_lane(store_dir, rig, dataroot="~/data/nuscenes_full", version="v1.0-trainval")
    r["ok"], r["problems"] (the first 40: token, message), r["stats"] (tier lengths, visible shares, frames without a tier 1, ...), r["lines"] (the printable summary)

Checks: the files and the index (complete, the keyframes of the version, no foreign or temporary file; the `masks` folder is the packed masks and belongs to the store); per file the keys, the frame, the
window (`window` and `rig` for the rectangular rig, none for lane_center), the sample token and the signature in the file, per-point arrays (tier 1 .. 3, tier_src 0 .. 2, vis / occ bool, the length of
`pts`, vis and occ exclusive, tier_src > 0 exactly where tier is 1), no NaN, every point of pieces / lines / areas / events inside the window, event tier and vis, `ego_path`, `tier_info` against the pieces.
"""
import gzip
import os
import pickle
import time
import zlib

import numpy as np

EPS = 1e-3


def _window(rig):
    from . import lane_gt
    spec = lane_gt.RIGS[rig]
    return (spec["window"] or (-50.0, 50.0, -50.0, 50.0)), spec["frame"]


def _inside(P, win):
    P = np.asarray(P, float).reshape(-1, 2)
    return not len(P) or (P[:, 0].min() >= win[0] - EPS and P[:, 0].max() <= win[1] + EPS and P[:, 1].min() >= win[2] - EPS and P[:, 1].max() <= win[3] + EPS)


def check_graph(tok, g, sig, rig):
    """the problem with one stored graph (a message) or the statistics of it (a dict)"""
    win, frame = _window(rig)
    for k in ("frame", "half", "pieces", "events", "lines", "areas", "params", "location", "sample", "signature", "ego_path", "tier_info"):
        if k not in g:
            return f"key {k} missing"
    if g["frame"] != frame or g["sample"] != tok or g["signature"] != sig:
        return f"frame {g['frame']} sample {g['sample']} signature {g['signature']} (want {frame}, {tok}, {sig})"
    if rig == "bev_fusion":
        if tuple(g.get("window", ())) != win or g.get("rig") != rig:
            return f"window {g.get('window')} rig {g.get('rig')}"
    elif "window" in g or "rig" in g or g["half"] != 50.0:
        return "a lane_center graph with window / rig / half != 50"
    n_pts, tl, tv = 0, {1: 0.0, 2: 0.0, 3: 0.0}, {1: 0.0, 2: 0.0, 3: 0.0}
    for q in g["pieces"]:
        P = np.asarray(q["pts"])
        n = len(P)
        for k, lo, hi in (("tier", 1, 3), ("tier_src", 0, 2)):
            a = np.asarray(q[k])
            if a.shape != (n,) or a.dtype != np.uint8 or (n and (a.min() < lo or a.max() > hi)):
                return f"piece {q['id']}: {k} {a.shape} {a.dtype}"
        for k in ("vis", "occ"):
            a = np.asarray(q[k])
            if a.shape != (n,) or a.dtype != bool:
                return f"piece {q['id']}: {k} {a.shape} {a.dtype}"
        if (np.asarray(q["vis"]) & np.asarray(q["occ"])).any():
            return f"piece {q['id']}: vis and occ at the same point"
        if not np.isfinite(P).all() or not _inside(P, win):
            return f"piece {q['id']}: points not finite or outside the window"
        if ((np.asarray(q["tier"]) == 1) != (np.asarray(q["tier_src"]) > 0)).any():
            return f"piece {q['id']}: tier_src > 0 must be exactly the tier-1 points"
        n_pts += n
        for t in (1, 2, 3):
            m = np.asarray(q["tier"]) == t
            tl[t] += 0.5 * m.sum()
            tv[t] += 0.5 * (m & np.asarray(q["vis"])).sum()
    for kind, items in g["lines"].items():
        for it in items:
            if not np.isfinite(it["pts"]).all() or not _inside(it["pts"], win):
                return f"line {kind} outside the window"
    for kind, polys in g["areas"].items():
        for p in polys:
            if not np.isfinite(p["exterior"]).all() or not _inside(p["exterior"], win):
                return f"area {kind} outside the window"
    for e in g["events"]:
        if "tier" not in e or "vis" not in e or e["tier"] not in (1, 2, 3) or not _inside(e["pos"], win):
            return f"event {e.get('type')}: tier / vis / position"
    ti = g["tier_info"]
    for t in (1, 2, 3):
        if abs(ti["length_m"][t] - tl[t]) > 1e-6 or abs(ti["visible_m"][t] - tv[t]) > 1e-6:
            return f"tier_info tier {t}: {ti['length_m'][t]} / {tl[t]}"
    ep = g["ego_path"]
    if not np.isfinite(ep["pts"]).all() or len(ep["pts"]) < 1 or ep["ahead_m"] < 0 or ep["back_m"] < 0:
        return "ego_path"
    return dict(n_pieces=len(g["pieces"]), n_pts=n_pts, t1=tl[1], t2=tl[2], t3=tl[3], v1=tv[1], v2=tv[2], v3=tv[3], ahead=ep["ahead_m"], back=ep["back_m"], boxes=ti["hiding_boxes"],
                ext=ti["extension_m"], bext=ti["back_extension_m"], events=len(g["events"]))


def audit_lane(store_dir, rig, dataroot=None, version="v1.0-trainval", scenes=None, limit=None, verbose=True):
    """audit the lane store `store_dir` of `rig` against the keyframes of `version` (all of them, or those of `scenes`); `dataroot` None = the store is checked against its own file list only.
    -> {"ok", "problems", "stats", "lines"}"""
    from .store import names, read_index
    problems = []

    def fail(tok, msg):
        if len(problems) < 40:
            problems.append((tok, msg))
            if verbose:
                print("BAD", tok, msg, flush=True)
    held = names(store_dir, ".pkl.gz")
    lines = []
    idx = read_index(store_dir)
    if dataroot is not None:
        from .suite import json_tables, scene_tokens
        root = os.path.expanduser(dataroot)
        want = {t for n in scenes for t in scene_tokens(n, version, root)} if scenes else set(json_tables(root, version)[1])
        lines.append(f"{store_dir}: {len(held)} files, {len(want)} keyframes asked for ({version}); missing {len(want - held)}, not asked for {len(held - want)}")
        if want - held or (not scenes and held - want):
            fail("store", f"files do not cover the keyframes: missing {len(want - held)}, foreign {len(held - want) if not scenes else 0}")
        order = sorted(held & want)
    else:
        lines.append(f"{store_dir}: {len(held)} files")
        order = sorted(held)
    odd = [f for f in os.listdir(store_dir) if not f.endswith(".pkl.gz") and f not in ("index.json", "masks")] if os.path.isdir(store_dir) else []
    if odd:
        fail("store", f"other files in the store: {odd[:5]}")
    if not idx.get("complete") or (dataroot is not None and not scenes and idx.get("keyframes") != len(want)):
        fail("store", f"the index does not say complete for these keyframes ({ {k: idx.get(k) for k in ('complete', 'keyframes', 'sets_complete')} })")
    lines.append(f"index: " + str({k: idx.get(k) for k in ("signature", "lane_version", "keyframes", "complete", "sets_complete", "commit", "dirty", "release")}))
    sig = idx.get("signature", os.path.basename(os.path.normpath(store_dir)))
    rows, t0 = [], time.time()
    for i, tok in enumerate(order[:limit]):
        try:
            with gzip.open(os.path.join(store_dir, tok + ".pkl.gz"), "rb") as f:
                g = pickle.load(f)
        except (EOFError, OSError, zlib.error, pickle.UnpicklingError) as e:              # a zero-byte or truncated file: a problem of this token, not a crash of the audit
            fail(tok, f"unreadable file ({type(e).__name__}: {e})")
            continue
        r = check_graph(tok, g, sig, rig)
        if isinstance(r, str):
            fail(tok, r)
        else:
            rows.append(r)
        if verbose and (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(order)} checked, {len(problems)} problems ({time.time() - t0:.0f} s)", flush=True)
    stats = {}
    if rows:
        A = {k: np.array([r[k] for r in rows], float) for k in rows[0]}
        stats = dict(frames=len(rows), tier1_mean_m=float(A["t1"].mean()), tier1_p5_m=float(np.percentile(A["t1"], 5)), tier2_mean_m=float(A["t2"].mean()), tier3_mean_m=float(A["t3"].mean()),
                     visible_tier1=float(A["v1"].sum() / max(A["t1"].sum(), 1)), visible_tier2=float(A["v2"].sum() / max(A["t2"].sum(), 1)), visible_tier3=float(A["v3"].sum() / max(A["t3"].sum(), 1)),
                     no_lane=int((A["n_pieces"] == 0).sum()), no_tier1=int((A["t1"] == 0).sum()), boxes_mean=float(A["boxes"].mean()), ext_ahead=float((A["ext"] > 0).mean()),
                     ext_behind=float((A["bext"] > 0).mean()), driven_under_15m=float((A["ahead"] < 15).mean()))
        lines.append(f"frames {stats['frames']}: tier-1 length mean {stats['tier1_mean_m']:.1f} m (5 % {stats['tier1_p5_m']:.1f}), tier 2 {stats['tier2_mean_m']:.1f} m, tier 3 {stats['tier3_mean_m']:.1f} m; "
                     f"visible share tier 1 {stats['visible_tier1']:.3f} / 2 {stats['visible_tier2']:.3f} / 3 {stats['visible_tier3']:.3f}")
        lines.append(f"frames without any lane {stats['no_lane']}, without tier 1 {stats['no_tier1']}, hiding boxes mean {stats['boxes_mean']:.1f}, map extension ahead in {stats['ext_ahead']:.3f} of the frames, "
                     f"behind in {stats['ext_behind']:.3f}, driven ahead < 15 m in {stats['driven_under_15m']:.3f}")
    ok = not problems
    lines.append(f"{'OK' if ok else str(len(problems)) + ' PROBLEMS'} ({len(rows)} files checked, {time.time() - t0:.0f} s)")
    if verbose:
        print("\n".join(lines), flush=True)
    return {"ok": ok, "problems": problems, "stats": stats, "lines": lines}
