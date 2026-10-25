"""D7 step 5-6 gate: the edited ground-truth scripts reproduce the stored b123 files on a check set.

    wsl bash tools/env/_py14.sh temp/loop19/check_b456_gt_repro.py setup     # token lists of three b123 scenes per part, in REVERSED scene order -> ~/exp_gcar_gtchk/feats
    bash temp/loop19/run_b456_gt.sh gtchk                                     # the same pipeline as for b456
    wsl bash tools/env/_py14.sh temp/loop19/check_b456_gt_repro.py compare   # every artefact against the stored b123 rows; PASS / FAIL

Whole scenes are taken so that the per-scene parts (previous frame, the future path of the route) are complete, as in b123; the reversed scene
order makes a pipeline that sorts by time instead of following the token list fail the check.
Exact equality is required for everything except the exact-distance map. That map: the stored b123 one was made with TF32 matrix products
(E235.chk2) and is wrong on some pixels in both directions, so it is no reference. Gate: (1) on every frame the new map never exceeds the
distance to the stored 0.5 m centre-line points (gt_pts_exact, a subset of the 0.1 m points, so an upper bound of the true distance) by more
than 0.01 px; (2) on four frames per part it equals a brute-force float64 distance to the 0.1 m points (rebuilt from the nuScenes map) within
0.01 px. The frames where the stored map differs from the new one by more than 0.05 px are counted (the TF32 damage), not gated.
"""
import json
import os
import sys

import numpy as np

H = os.path.expanduser("~")
B, C = f"{H}/exp_gcar_b123", f"{H}/exp_gcar_gtchk"
SCENES = {"train": [12, 7, 3], "val": [9, 5, 1]}  # scene numbers inside the b123 parts (poses 'scene'), taken in this order


def blocks(part):
    sc = np.load(f"{B}/poses_{part}.npz")["scene"]
    return [np.nonzero(sc == s)[0] for s in SCENES[part]]


def rows_of(part):
    return np.concatenate(blocks(part))


def same(a, b, path=""):
    """deep equality of the pickled ground truth (dicts, lists, tuples, arrays, numbers); returns the first difference or None"""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return None if a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b, equal_nan=a.dtype.kind == "f") else f"{path}: array differs"
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return f"{path}: keys {sorted(set(a) ^ set(b))}"
        for k in a:
            d = same(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = same(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return None
    return None if a == b else f"{path}: {a!r} vs {b!r}"


def bound_px(pts):
    """distance (px, capped at 8, fp16 like the map) of every pixel centre to the 0.5 m centre-line points"""
    from scipy.spatial import cKDTree

    cen = -50.0 + (np.arange(200) + 0.5) * 0.5
    gx, gy = np.meshgrid(cen, cen, indexing="ij")
    d = cKDTree(np.asarray(pts, np.float64)).query(np.stack([gx.reshape(-1), gy.reshape(-1)], -1))[0] if len(pts) else np.full(40000, np.inf)
    return np.minimum(d / 0.5, 8.0).astype(np.float16).astype(np.float32).reshape(200, 200)


_NUSC = {}


def truth_px(token):
    """brute-force float64 distance (px, capped at 8, fp16) of every pixel centre to the 0.1 m centre-line points, rebuilt like make_dist_gt.py"""
    from nuscenes.map_expansion.map_api import NuScenesMap
    from nuscenes.nuscenes import NuScenes

    sys.path.insert(0, os.path.join(os.environ["BFLC_ROOT"], "training/data/prep"))
    from centerline_gt import build_lane_polylines, lidar_pose

    os.environ["GT_FRAME"] = "ego"
    if "n" not in _NUSC:
        _NUSC["n"] = NuScenes("v1.0-trainval", f"{H}/data/nuscenes_full", verbose=False)
    nusc = _NUSC["n"]
    s = nusc.get("sample", token)
    loc = nusc.get("log", nusc.get("scene", s["scene_token"])["log_token"])["location"]
    if loc not in _NUSC:
        _NUSC[loc] = build_lane_polylines(NuScenesMap(dataroot=f"{H}/data/nuscenes_full", map_name=loc), 0.5)
    polys, bb = _NUSC[loc]
    t, R = lidar_pose(nusc, s)
    near = (bb[:, 2] > t[0] - 80) & (bb[:, 0] < t[0] + 80) & (bb[:, 3] > t[1] - 80) & (bb[:, 1] < t[1] + 80)
    fine = []
    for i in np.nonzero(near)[0]:
        q = (polys[i] - t) @ R
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
        u = np.arange(0, d[-1] + 1e-6, 0.1) if d[-1] >= 1e-6 else np.zeros(1)
        fine.append(np.stack([np.interp(u, d, q[:, 0]), np.interp(u, d, q[:, 1])], 1))
    fine = np.concatenate(fine)
    fine = fine[(np.abs(fine) < 56).all(1)].astype(np.float32).astype(np.float64)
    cen = -50.0 + (np.arange(200) + 0.5) * 0.5
    gx, gy = np.meshgrid(cen, cen, indexing="ij")
    pix = np.stack([gx.reshape(-1), gy.reshape(-1)], -1)
    out = np.empty(len(pix))
    for a in range(0, len(pix), 2000):
        out[a:a + 2000] = np.sqrt(((pix[a:a + 2000, None, :] - fine[None]) ** 2).sum(-1)).min(1)
    return np.minimum(out / 0.5, 8.0).astype(np.float16).astype(np.float32).reshape(200, 200)


def setup():
    os.makedirs(f"{C}/feats", exist_ok=True)
    for part in ("train", "val"):
        toks = json.load(open(f"{B}/feats/{part}_tokens.json"))
        json.dump([toks[i] for i in rows_of(part)], open(f"{C}/feats/{part}_tokens.json", "w"))
        print(part, "scenes", SCENES[part], "rows", [f"{b[0]}-{b[-1]}" for b in blocks(part)], len(rows_of(part)))


def compare():
    ok = True

    def report(name, diff):
        nonlocal ok
        ok &= diff is None
        print(f"  {name:28} {'identical' if diff is None else 'DIFFERS: ' + diff}")

    crn_tok = {p: json.load(open(f"{H}/exp_crn_b123/feats/{p}_tokens.json")) for p in ("train", "val")}
    for part in ("train", "val"):
        r = rows_of(part)
        toks = json.load(open(f"{B}/feats/{part}_tokens.json"))
        assert json.load(open(f"{C}/feats/{part}_tokens.json")) == [toks[i] for i in r]
        print(f"{part}: {len(r)} frames in {len(SCENES[part])} scenes, reversed order")
        import pickle
        for f in ("graph_gt", "route_gt", "sd", "route_pts"):
            ref = pickle.load(open(f"{B}/{f}_{part}.pkl", "rb"))
            new = pickle.load(open(f"{C}/{f}_{part}.pkl", "rb"))
            report(f"{f}_{part}.pkl", same([ref[i] for i in r], new, f))
        P0, P1 = np.load(f"{B}/poses_{part}.npz"), np.load(f"{C}/poses_{part}.npz")
        for k in ("t", "R", "loc", "dt", "ts"):
            report(f"poses.{k}", same(P0[k][r], P1[k], k))
        prev_ref, scene_ref, start = [], [], 0
        for n, b in enumerate(blocks(part)):  # the previous frame re-indexed into the check set, block by block
            p = P0["prev"][b]
            prev_ref.append(np.where(p >= 0, p - b[0] + start, -1))
            scene_ref.append(np.full(len(b), n))
            start += len(b)
        report("poses.prev (re-indexed)", same(np.concatenate(prev_ref), P1["prev"], "prev"))
        report("poses.scene (re-numbered)", same(np.concatenate(scene_ref), P1["scene"], "scene"))
        g0, g1 = np.load(f"{B}/gtK_{part}.npz"), np.load(f"{C}/gtK_{part}.npz")
        for k in ("pts", "valid", "adj", "orig"):
            report(f"gtK.{k}", same(g0[k][r], g1[k], k))
        report(f"dir_{part}.npy", same(np.load(f"{B}/dir_{part}.npy", mmap_mode="r")[r], np.load(f"{C}/dir_{part}.npy"), "dir"))
        report(f"route_heat_{part}.npy", same(np.load(f"{B}/route_heat_{part}.npy", mmap_mode="r")[r], np.load(f"{C}/route_heat_{part}.npy"), "heat"))
        pos = {t: i for i, t in enumerate(crn_tok[part])}
        rc = np.array([pos[toks[i]] for i in r])
        pts0 = np.load(f"{H}/exp_crn_b123/feats/{part}_gt_pts_exact.npy", allow_pickle=True)[rc]
        pts1 = np.load(f"{C}/feats/{part}_gt_pts_exact.npy", allow_pickle=True)
        report("gt_pts_exact", same(list(pts0), list(pts1), "pts"))
        d0 = np.load(f"{H}/exp_crn_b123/feats/{part}_gt_dist_exact.npy", mmap_mode="r")[rc].astype(np.float32)
        d1 = np.load(f"{C}/feats/{part}_gt_dist_exact.npy").astype(np.float32)
        over_new = max(float((d1[k] - bound_px(pts1[k])).max()) for k in range(len(r)))
        damaged = [k for k in range(len(r)) if np.abs(d0[k] - d1[k]).max() > 0.05]
        sample = np.linspace(0, len(r) - 1, 4).astype(int)
        err_truth = max(float(np.abs(d1[k] - truth_px(toks[r[k]])).max()) for k in sample)
        good = over_new <= 0.01 and err_truth <= 0.01
        ok &= good
        print(f"  {'gt_dist_exact':28} new map over the 0.5 m-point bound by at most {over_new:.4f} px (all {len(r)} frames); max |new - float64 truth| "
              f"{err_truth:.4f} px (frames {list(sample)}) -> {'ok' if good else 'FAIL'};  stored b123 map differs by > 0.05 px in {len(damaged)} of {len(r)} frames (TF32)")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(setup() if sys.argv[1:] == ["setup"] else compare())
