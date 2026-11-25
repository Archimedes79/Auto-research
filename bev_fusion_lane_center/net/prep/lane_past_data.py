"""E368 (stage S2 of the token champion, D113): the data of the cross-attention to the PAST keyframe: the previous keyframe of every cached frame, the ego pose between the two, the sequence loader, the fixture of the pose and the step-0 measurement.
Derived data (not in git): the extension files next to the arrays of the E365 cache directories (`lane_tokcache.py`); the code, its selftest and `past.json` are the record.

    wsl bash tools/env/_py14.sh lane_past_data.py build   [--cache <split dir> | --all]                       # prev.npy, rel_R.npy, rel_t.npy, past.json next to the arrays of one cache directory / of all eight (idempotent, atomic)
    wsl bash tools/env/_py14.sh lane_past_data.py check                                                        # every built directory: extension current, frames / scene starts of the data, VAL = the rows of data/splits.json
    wsl bash tools/env/_py14.sh lane_past_data.py fixture --split val [--n-pairs 200] [--seed 0] [--ident e292_a] [--out FILE]   # the pose against the GT lane points and the B' tokens: PASS / FAIL + numbers; the wrong conventions must FAIL
    wsl bash tools/env/_py14.sh lane_past_data.py potential --split val --seed 0 [--n N] [--out FILE]        # step 0 of the E368 card: what the past could give (JSON)
    wsl bash tools/env/_py14.sh lane_past_data.py selftest                                                   # no cache needed; tests: net/tests/test_lane_past.py (the hand mutants)
    ds, dl = loader("train", seed=0, bs=4, workers=4, K_seq=3)      # TRAIN: chains of K_seq consecutive keyframes, shuffled; "val": the STREAM (one frame per item, scene order, shuffle False, bs 1: the state is carried by the caller)
    batch["frames"]   a list of n ordinary batches of `lane_ptok_data.collate` (n = K_seq, 1 in a stream): frames[k][b] = frame k of sequence b
    batch["rel_R"] (B, n, 2, 2), ["rel_t"] (B, n, 2), ["has_prev"] (B, n) bool: entry k = the lane frame of the PREVIOUS keyframe of frame k in that of frame k (p_cur = R p_prev + t) and whether frame k has one; in a chain the entries k >= 1 link the
                      chain's own frames (has_prev true), entry 0 is the frame before the chain (the trainer's step 1 never uses it); batch["idx"] (B, n) the cache indices

Pose: `poses_<part>.npz` (`p16_data.D`; keys prev, R, t, ts, scene; p_global = t + R p_lane, rows of the part) and `lane_seq_data.rel_pose` (R = Rc^T Rp, t = Rc^T (tp - tc)). The cache order IS the stream order (scene, time) on TRAIN and VAL, so the previous keyframe of
a frame is the cache row right before it or none (asserted): prev.npy (F,) int32 = that cache index or -1, rel_R (F, 2, 2) / rel_t (F, 2) float32 (identity / zero where prev = -1). TEST guard: `poses_val.npz` also holds the TEST rows (1706 rows, 873 VAL + 833 TEST):
only the rows of the cache's frames.json (and the previous rows they name, which must be in it) are ever read, asserted against `data/splits.json['val']` (873 rows, 22 scene starts; TRAIN 8414 rows, 212 scene starts). past.json names the poses file (sha256), the convention, the
counts of frames with / without a past and the sha256 of frames.json (a changed cache is refused by the loader).
The fixture (before any training): the GT lane points of t-1 (`lane_gt.load(token)['pieces'][i]['pts']`) warped by the stored pose land on those of t (points inside the 45 m box, within 6 m of a point of t: p90 <= 0.10 m, p99 <= 0.25 m, the tangents p90 <= 5 degrees),
the warped delivered B' tokens (valid, p >= 0.5) of t-1 have a delivered token of t within p90 <= 1.5 m; the identity pose, the inverse pose, the transposed rotation and a previous frame shifted by one must FAIL the point test.
(The loader fixture through `jpast.make_past` left with the S2 block: archive tag archive/s2-s3-blocks-2026-11-18.)
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub_ in ("training/data", "net/prep", "net/model", "net/loss", "evaluation"):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import lane_ptok_data as LP  # noqa: E402
import lane_tokcache as LC  # noqa: E402

FORMAT = 1
FILES = ("prev", "rel_R", "rel_t")
CONVENTION = "p_cur = R p_prev + t with R = Rc^T Rp, t = Rc^T (tp - tc) (lane_seq_data.rel_pose; lane frame x right / y forward; poses_<part>.npz p_global = t + R p_lane); identity / zero where there is no previous keyframe"
EXPECT = dict(val=(873, 22), train=(8414, 212))                                   # frames and scene starts of the cache directories (data/splits.json, checked 2026-11-17)
TOL = dict(p90=0.10, p99=0.25, head_p90_deg=5.0, tok_p90=1.5)                    # the fixture's tolerances, named before the build (E368 card, section 1)
BOX, NEAR = 45.0, 6.0                                                            # the fixture looks at warped points inside the 45 m box with a point of t within 6 m
CONVENTIONS = ("warp", "identity", "inverse", "transposed", "shifted")           # the right pose and the wrong ones the fixture must reject
RAD, COS = 1.5, 0.5                                                              # `potential`: a token is near a GT node / supported by a past token within 1.5 m and a heading cosine >= 0.5 (the card's step 0)
BINS = ((0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5))
TOPO = (0, 1, 2, 3, 4, 5)                                                        # GT node flags of a topology node: stop, exit, split, merge, start, end (`lane_tokcache.FLAGS`)


# ------------------------------------------------------------------------------------------------------------------------------ files
def _atomic(path, write):
    root, ext = os.path.splitext(path)
    tmp = f"{root}.{os.getpid()}.tmp{ext}"                                           # prev.<pid>.tmp.npy (np.save keeps a name that ends in .npy)
    write(tmp)
    os.replace(tmp, path)


def _save_npy(path, arr):
    _atomic(path, lambda tmp: np.save(tmp, arr))


def _save_json(path, obj):
    def w(tmp):
        with open(tmp, "w") as fh:
            json.dump(obj, fh, indent=1, sort_keys=True)
    _atomic(path, w)


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()[:16]


def pose_path(split):
    import p16_data as PD  # heavy (scipy, graph_eval): only the build and the checks need the poses
    return os.path.join(PD.D, f"poses_{split}.npz")


def link(pose, rows):
    """the previous keyframe and the pose of every cache frame -> (prev (F,) int32: the cache index of the previous keyframe or -1, rel_R (F, 2, 2), rel_t (F, 2) float32, p_cur = R p_prev + t; identity / zero without a previous keyframe).
    Only the rows of `rows` are read from `pose` (the VAL poses file also holds the TEST rows) and the previous row of a frame must be one of them"""
    import lane_seq_data as LS
    rows = np.asarray(rows, np.int64)
    F = len(rows)
    idx_of = {int(r): i for i, r in enumerate(rows)}
    p_row = np.asarray(pose["prev"])[rows]
    prev, R, t = np.full(F, -1, np.int32), np.tile(np.eye(2, dtype=np.float32), (F, 1, 1)), np.zeros((F, 2), np.float32)
    for i, pr in enumerate(p_row):
        if pr < 0:
            continue
        assert int(pr) in idx_of, f"frame {i} (row {rows[i]}): the previous keyframe (row {pr}) is not in the cache"
        prev[i] = idx_of[int(pr)]
        R[i], t[i] = LS.rel_pose(pose["R"][rows[i]], pose["t"][rows[i]], pose["R"][pr], pose["t"][pr])
    return prev, R, t


def check_rows(split, rows, splits_val=None, expect=EXPECT):
    """the TEST guard: the rows of a VAL cache are exactly `data/splits.json['val']`, those of TRAIN are distinct; the frame count is the one of the data (`expect`, None = not checked)"""
    assert len(set(rows)) == len(rows), "a row twice in the cache"
    if split == "val":
        sv = [int(r) for r in (splits_val if splits_val is not None else json.load(open(os.path.join(ROOT, "data", "splits.json")))["val"])]
        assert [int(r) for r in rows] == sv, "the rows of the VAL cache are not data/splits.json['val'] (the poses file also holds TEST rows)"
    if expect:
        assert len(rows) == expect[split][0], f"{len(rows)} rows, expected {expect[split][0]}"


def check_stream(pose, rows, prev, expect=EXPECT, split=None):
    """the cache order is the stream order (scene, time), the previous keyframe is the row right before, the scene starts are the frames without one"""
    import lane_seq_data as LS
    so = LS.stream_order(pose, rows)
    assert [int(r) for r, _ in so] == [int(r) for r in rows], "the cache order is not the stream order (scene, time)"
    assert [bool(x) for _, x in so] == [bool(p < 0) for p in prev], "the scene starts of the stream order are not the frames without a previous keyframe"
    has = prev >= 0
    assert (prev[has] == (np.arange(len(prev)) - 1)[has]).all(), "a previous keyframe is not the cache row right before the frame"
    if expect:
        assert int((~has).sum()) == expect[split][1], f"{int((~has).sum())} scene starts, expected {expect[split][1]}"


def build(cache_dir, pose_file=None, check=True):
    """write prev.npy, rel_R.npy, rel_t.npy and past.json next to the arrays of the cache directory `cache_dir` (<run>/<split>); idempotent (the same bytes), every file written whole and renamed, past.json last.
    `check` False (the selftest's toy cache only) skips the data fingerprint of VAL / TRAIN -> the written past.json"""
    d = os.path.normpath(cache_dir)
    split = os.path.basename(d)
    assert split in ("train", "val"), f"{d} is not a <run>/<split> directory"
    fr = json.load(open(os.path.join(d, "frames.json")))
    meta = json.load(open(os.path.join(d, "meta.json")))
    assert meta["split"] == split and os.path.exists(os.path.join(d, "manifest.json")), f"{d} is not a finished cache of the split {split}"
    rows = fr["rows"]
    pf = pose_file or pose_path(split)
    pose = dict(np.load(pf))
    if check:
        check_rows(split, rows)
    prev, R, t = link(pose, rows)
    check_stream(pose, rows, prev, EXPECT if check else None, split)
    for n, a in zip(FILES, (prev, R, t)):
        _save_npy(os.path.join(d, n + ".npy"), a)
    info = dict(format=FORMAT, split=split, frames=len(rows), with_past=int((prev >= 0).sum()), without_past=int((prev < 0).sum()), poses=os.path.basename(pf), poses_sha=_sha(pf), convention=CONVENTION, frames_sha=_sha(os.path.join(d, "frames.json")),
                test_guard="only the rows of frames.json are read from the poses file (the VAL file also holds the TEST rows); a VAL cache is data/splits.json['val'] exactly", prev_is_the_row_before=True)
    _save_json(os.path.join(d, "past.json"), info)
    return info


def cache_dirs(cache=LC.CACHE):
    """the finished cache directories <run>/<split> of the root (the limited `_first<N>` builds are not in it)"""
    return sorted(p for p in glob.glob(os.path.join(cache, f"b*_{LC.VERSION}", "*")) if os.path.basename(p) in ("train", "val") and "_first" not in p and os.path.exists(os.path.join(p, "manifest.json")))


def check_built(cache=LC.CACHE, log=print):
    """every finished cache directory has its extension, current (the sha of frames.json), with the frames and scene starts of the data and, for VAL, exactly the rows of data/splits.json['val'] -> the number of directories checked"""
    dirs = cache_dirs(cache)
    assert len(dirs) >= 1, f"no finished cache directory under {cache}"
    for d in dirs:
        ext, split = load_ext(d), os.path.basename(d)
        rows = json.load(open(os.path.join(d, "frames.json")))["rows"]
        check_rows(split, rows)
        assert (len(ext["prev"]), int((ext["prev"] < 0).sum())) == EXPECT[split] and ext["info"]["with_past"] == EXPECT[split][0] - EXPECT[split][1], f"{d}: {ext['info']}"
        log(f"ok {os.path.relpath(d, ROOT)}: {len(ext['prev'])} frames, {int((ext['prev'] < 0).sum())} scene starts")
    return len(dirs)


def load_ext(d):
    """the extension of a cache directory, checked against its frames.json (a rebuilt cache is refused) -> dict(prev, rel_R, rel_t, info)"""
    pj = os.path.join(d, "past.json")
    assert os.path.exists(pj), f"{d} has no past.json: run `lane_past_data.py build --cache {d}`"
    info = json.load(open(pj))
    assert info["format"] == FORMAT and info["frames_sha"] == _sha(os.path.join(d, "frames.json")), f"{d}: past.json belongs to another cache or format: rebuild"
    out = {n: np.load(os.path.join(d, n + ".npy")) for n in FILES}
    F = info["frames"]
    assert out["prev"].shape == (F,) and out["rel_R"].shape == (F, 2, 2) and out["rel_t"].shape == (F, 2), f"{d}: the extension arrays do not have {F} frames"
    return dict(out, info=info)


# ------------------------------------------------------------------------------------------------------------------------------ sequences
def chain_starts(prev, K_seq):
    """the starts s of the runs s, s + 1, .., s + K_seq - 1 of consecutive cache frames in which every frame after the first has its previous keyframe (the row right before it): no chain crosses a scene start"""
    has = np.asarray(prev) >= 0
    return [s for s in range(len(prev) - K_seq + 1) if has[s + 1: s + K_seq].all()]


def scene_ids(prev):
    """(F,) int: the scene of every frame of the stream (a new scene at every frame without a previous keyframe)"""
    return np.cumsum(np.asarray(prev) < 0) - 1


class PastData(torch.utils.data.Dataset):
    """the frames of `lane_ptok_data.PTokData` + the poses of the extension; stream False (TRAIN): items of K_seq consecutive frames (`chain_starts`), True (VAL): one frame per item in the cache (= stream) order"""

    def __init__(self, split="train", seed=0, ident="e292_a", K=None, K_seq=3, token_drop=0.0, cache=LC.CACHE, dir=None, stream=None):
        self.tok = LP.PTokData(split, seed, ident, K, token_drop, cache, dir)
        self.dir = self.tok.dir
        ext = load_ext(self.dir)
        assert len(ext["prev"]) == len(self.tok), f"{self.dir}: {len(ext['prev'])} poses for {len(self.tok)} frames"
        self.prev, self.rel_R, self.rel_t, self.info = ext["prev"], ext["rel_R"], ext["rel_t"], ext["info"]
        self.stream = (split != "train") if stream is None else bool(stream)
        self.K_seq = 1 if self.stream else int(K_seq)
        self.items = [[i] for i in range(len(self.tok))] if self.stream else [list(range(s, s + self.K_seq)) for s in chain_starts(self.prev, self.K_seq)]
        self.scene = scene_ids(self.prev)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, n):
        ix = self.items[n]
        return dict(frames=[self.tok[i] for i in ix], rel_R=self.rel_R[ix], rel_t=self.rel_t[ix], has_prev=self.prev[ix] >= 0, idx=np.array(ix, np.int64))


def collate(b):
    """a list of items -> dict(frames = a list of n `lane_ptok_data.collate` batches (frames[k][b] = frame k of item b), rel_R (B, n, 2, 2), rel_t (B, n, 2), has_prev (B, n) bool, idx (B, n))"""
    n = len(b[0]["frames"])
    return dict(frames=[LP.collate([o["frames"][k] for o in b]) for k in range(n)], **{k: torch.from_numpy(np.stack([o[k] for o in b])) for k in ("rel_R", "rel_t", "has_prev", "idx")})


def loader(split="train", seed=0, bs=4, workers=4, K_seq=3, token_drop=0.0, ident="e292_a", K=None, shuffle=None, cache=LC.CACHE, dir=None, stream=None):
    ds = PastData(split, seed, ident, K, K_seq, token_drop, cache, dir, stream)
    shuffle = (split == "train") if shuffle is None else shuffle
    return ds, torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers, collate_fn=collate, drop_last=shuffle, persistent_workers=workers > 0, prefetch_factor=2 if workers > 0 else None)


# ------------------------------------------------------------------------------------------------------------------------------ the fixture
def warp_pts(P, R, t, how="warp"):
    """the points P (n, 2) of the previous frame as `how` puts them into the current one: warp = R p + t (the pose of the extension), identity, inverse = R^T (p - t), transposed = R^T p + t; shifted is a pair choice (`pairs`), its points are warped"""
    P, R, t = np.asarray(P, float), np.asarray(R, float), np.asarray(t, float)
    if how in ("warp", "shifted"):
        return P @ R.T + t
    if how == "identity":
        return P
    if how == "inverse":
        return (P - t) @ R
    if how == "transposed":
        return P @ R + t
    raise ValueError(how)


def warp_dirs(T, R, how="warp"):
    """the tangents / headings of the previous frame in the current one: only the rotation of `warp_pts` (the identity keeps them)"""
    return np.asarray(T, float) if how == "identity" else np.asarray(T, float) @ (np.asarray(R, float).T if how in ("warp", "shifted") else np.asarray(R, float))


def tangents(pts):
    """(n, 2) unit tangents along a polyline (central differences; a polyline of one point has none)"""
    g = np.gradient(np.asarray(pts, float), axis=0)
    return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)


def stats(x):
    x = np.asarray(x, float)
    return dict(n=int(len(x)), median=float(np.median(x)), p90=float(np.percentile(x, 90)), p99=float(np.percentile(x, 99))) if len(x) else dict(n=0, median=float("nan"), p90=float("nan"), p99=float("nan"))


def point_pass(s, tol=TOL):
    """the point test: p90 and p99 of the distances within the tolerance (False for an empty set)"""
    return s["n"] > 0 and s["p90"] <= tol["p90"] and s["p99"] <= tol["p99"]


def pair_errors(P0, T0, P1, T1, R, t, how="warp"):
    """one pair: the points P0 (tangents T0) of t-1 put into frame t by `how`, those inside the 45 m box with a point of P1 (tangents T1) within 6 m -> (distances, angles in degrees between the tangents)"""
    from scipy.spatial import cKDTree
    W = warp_pts(P0, R, t, how)
    inside = (np.abs(W) < BOX).all(1)
    if not inside.any() or len(P1) == 0:
        return np.zeros(0), np.zeros(0)
    d, k = cKDTree(P1).query(W[inside])
    near = d < NEAR
    cs = (warp_dirs(T0, R, how)[inside][near] * T1[k[near]]).sum(1)
    return d[near], np.degrees(np.arccos(np.clip(cs, -1.0, 1.0)))


def token_errors(A, B, R, t, how="warp"):
    """the delivered tokens A (n, 2) of t-1 warped into frame t and the delivered tokens B of t: inside the box, distance to the nearest token of B below 6 m"""
    from scipy.spatial import cKDTree
    W = warp_pts(A, R, t, how)
    W = W[(np.abs(W) < BOX).all(1)]
    if len(W) == 0 or len(B) == 0:
        return np.zeros(0)
    d, _ = cKDTree(B).query(W)
    return d[d < NEAR]


def pairs(prev, n_pairs, seed=0, shift=0):
    """n_pairs (previous index, index) pairs of frames with a past, drawn without replacement by default_rng(seed); shift 1: the previous frame is the one BEFORE the stored previous (the mutant 'prev_idx shifted by one frame': its pose is still the stored one)"""
    cur = [i for i in range(len(prev)) if prev[i] >= 0 and (shift == 0 or prev[prev[i]] >= 0)]
    sel = np.random.default_rng(seed).choice(len(cur), min(n_pairs, len(cur)), replace=False)
    out = [(int(prev[cur[s]]), cur[s]) for s in sel]
    return [(int(prev[ip]), ic) for ip, ic in out] if shift else out


def gt_points(g):
    """the lane GT graph of a frame -> (points (n, 2), tangents (n, 2)) of all pieces (a piece with one point has no tangent and is left out)"""
    ps = [np.asarray(p["pts"], float) for p in g["pieces"] if len(p["pts"]) >= 2]
    return (np.concatenate(ps), np.concatenate([tangents(p) for p in ps])) if ps else (np.zeros((0, 2)), np.zeros((0, 2)))


def pose_of(ext, i):
    """the stored pose of frame i: (R (2, 2), t (2,)) with p_cur = R p_prev + t, p_prev a point in the lane frame of the PREVIOUS keyframe of frame i, p_cur the same point in the lane frame of frame i (the one place the fixture and `potential` read it)"""
    return ext["rel_R"][i], ext["rel_t"][i]


def fixture_checks(points, headings, tokens, hows=CONVENTIONS, tol=TOL):
    """the rule of the fixture on the stats per convention (dicts of `stats`) -> name -> bool: the right pose within every tolerance, every wrong convention FAILS the point test"""
    ok = {"points warp within tolerance": point_pass(points["warp"], tol),
          "headings warp p90 within tolerance": headings["warp"]["n"] > 0 and headings["warp"]["p90"] <= tol["head_p90_deg"],
          "tokens warp p90 within tolerance": tokens["warp"]["n"] > 0 and tokens["warp"]["p90"] <= tol["tok_p90"]}
    for how in hows:
        if how != "warp":
            ok[f"{how} fails the point test"] = not point_pass(points[how], tol)
    return ok


def fixture(split="val", n_pairs=200, seed=0, ident="e292_a", cache=LC.CACHE, dir=None, hows=CONVENTIONS, log=print):
    """the module docstring's fixture on n_pairs pairs of the stored extension -> dict(split, n_pairs, points / headings / tokens: per convention the stats, checks: name -> bool, passed); passed = the right pose within the tolerances and every wrong convention
    FAILS the point test. The GT lane store of this checkout and the cache of B' seed `seed`"""
    import lane_gt_data as LD
    lane_gt, _ = LD.lane_store()
    d = dir or LP.cache_dir(split, seed, ident, None, cache)
    ext, c = load_ext(d), LC.TokCache(d)
    pts = {}

    def gt(i):
        if i not in pts:
            pts[i] = gt_points(lane_gt.load(c.tokens[i]))
        return pts[i]

    def deliv(i):
        m = np.asarray(c.f["valid"][i]) & (np.asarray(c.f["e_p"][i], float) >= 0.5)
        return np.asarray(c.f["pos"][i], float)[m]

    err = {h: ([], [], []) for h in hows}
    for how in hows:
        for ip, ic in pairs(ext["prev"], n_pairs, seed, shift=int(how == "shifted")):
            R, t = pose_of(ext, ic)
            (P0, T0), (P1, T1) = gt(ip), gt(ic)
            if len(P0) and len(P1):
                dd, aa = pair_errors(P0, T0, P1, T1, R, t, how)
                err[how][0].append(dd)
                err[how][1].append(aa)
            err[how][2].append(token_errors(deliv(ip), deliv(ic), R, t, how))
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)  # noqa: E731
    res = dict(split=split, n_pairs=n_pairs, seed=seed, cache=os.path.relpath(d, ROOT), tolerances=TOL, points={}, headings={}, tokens={}, checks={})
    for how in hows:
        res["points"][how], res["headings"][how], res["tokens"][how] = (stats(cat(e)) for e in err[how])
    ok = res["checks"] = fixture_checks(res["points"], res["headings"], res["tokens"], hows)
    res["passed"] = all(ok.values())
    log(f"fixture {split} {len(ext['prev'])} frames, {n_pairs} pairs of {d}")
    for how in hows:
        p, h, k = res["points"][how], res["headings"][how], res["tokens"][how]
        log(f"  {how:<11} points n {p['n']:6d} median {p['median']:.3f} p90 {p['p90']:.3f} p99 {p['p99']:.3f} m | heading p90 {h['p90']:.2f} deg | B' tokens n {k['n']:5d} median {k['median']:.2f} p90 {k['p90']:.2f} m")
    for k, v in ok.items():
        log(f"  {'ok  ' if v else 'FAIL'} {k}")
    log("PASS" if res["passed"] else "FAIL")
    return res


# ------------------------------------------------------------------------------------------------------------------------------ step 0: what the past could give
def near_of(P, D, gpos, gdir):
    """(n, M) bool: the tokens P / D within RAD of a GT node with a heading cosine >= COS"""
    if len(P) == 0 or len(gpos) == 0:
        return np.zeros((len(P), len(gpos)), bool)
    return (np.linalg.norm(P[:, None] - gpos[None], axis=2) <= RAD) & (D @ gdir.T >= COS)


def supported(P, D, A, B):
    """(n,) bool: a token of P / D has a token of A / B within RAD and a heading cosine >= COS"""
    if len(A) == 0 or len(P) == 0:
        return np.zeros(len(P), bool)
    return ((np.linalg.norm(P[:, None] - A[None], axis=2) <= RAD) & (D @ B.T >= COS)).any(1)


def frame_counts(pos, dirv, valid, ep, gpos, gdir, gfl, sup):
    """the counters of ONE frame that has a past: pos, dirv, valid, ep = the tokens of t (K, ...), gpos / gdir / gfl (M, ...) the GT nodes of t with their flags, sup = dict name -> (positions, headings) of the delivered tokens of t-1
    as `name` puts them into frame t (pose / identity / random: the card's nulls) -> dict of counters (summed by `potential`; `rows` = the weak tokens (p, true, supported by the pose, novel)). A weak token = valid and p < 0.5, true = near a GT node, novel =
    near a GT node that NO delivered token of t found, duplicate = near found GT nodes only, false = near none"""
    M = len(gpos)
    deliv = valid & (ep >= 0.5)
    weak = valid & ~deliv
    out = dict(rows=np.zeros((0, 4)))
    cnt = lambda k, v: out.__setitem__(k, out.get(k, 0) + int(v))  # noqa: E731
    near = lambda m: near_of(pos[m], dirv[m], gpos, gdir)  # noqa: E731
    if M:
        found = near(deliv).any(0) if deliv.any() else np.zeros(M, bool)
        miss = ~found
        has_weak = near(weak).any(0) if weak.any() else np.zeros(M, bool)
        pp, pd = sup["pose"]
        past = near_of(pp, pd, gpos, gdir).any(0)                                           # a delivered warped token of t-1 is near the GT node
        topo = gfl[:, list(TOPO)].any(1)
        se = gfl[:, [4, 5]].any(1)
        for k, v in dict(gt=M, found=found.sum(), missed=miss.sum(), missed_weak=(miss & has_weak).sum(), missed_absent=(miss & ~has_weak).sum(), missed_past=(miss & past).sum(), missed_weak_past=(miss & has_weak & past).sum(),
                         missed_weak_past_topo=(miss & has_weak & past & topo).sum(), missed_weak_past_startend=(miss & has_weak & past & se).sum(), missed_topo=(miss & topo).sum(), missed_startend=(miss & se).sum(), gt_topo=topo.sum()).items():
            cnt(k, v)
    for name, sel in (("w", weak), ("d", deliv)):
        if not sel.any():
            continue
        tr = near(sel).any(1) if M else np.zeros(sel.sum(), bool)
        sp = supported(pos[sel], dirv[sel], *sup["pose"])
        for lab, key in ((True, "t"), (False, "f")):
            cnt(f"{key}{name}", (tr == lab).sum())
            cnt(f"{key}{name}_s", (sp & (tr == lab)).sum())
        if name == "w":
            novel = (near(sel) & miss[None]).any(1) if M else np.zeros(sel.sum(), bool)
            out["rows"] = np.stack([ep[sel], tr.astype(float), sp.astype(float), novel.astype(float)], 1)
            if M:
                nr = near(sel)
                dup = nr.any(1) & ~novel
                for role, m in (("novel", novel), ("dup", dup), ("false", ~nr.any(1))):
                    cnt(f"{role}_n", m.sum())
                    for nm, (A, B) in sup.items():
                        cnt(f"{role}_{nm}", (supported(pos[sel], dirv[sel], A, B) & m).sum())
    return out


def bin_table(W):
    """the weak tokens W (n, 4) = existence p, true (near a GT node), supported by the warped past, novel (near a GT node no delivered token found) as 0 / 1 -> one row per existence bin [lo, hi): n, the true share, the support rate of the true / of the false
    tokens, the novel share among the supported / among the unsupported tokens (the support separates novel from false only if these two differ)"""
    r = lambda a, b: float(a) / max(float(b), 1.0)  # noqa: E731
    out = []
    for lo, hi in BINS:
        m = (W[:, 0] >= lo) & (W[:, 0] < hi)
        t_, f_ = m & (W[:, 1] > 0.5), m & (W[:, 1] < 0.5)
        s_, u_ = m & (W[:, 2] > 0.5), m & (W[:, 2] < 0.5)
        out.append(dict(lo=lo, hi=hi, n=int(m.sum()), true_share=r(t_.sum(), m.sum()), supported_if_true=float(W[t_, 2].mean()) if t_.any() else float("nan"), supported_if_false=float(W[f_, 2].mean()) if f_.any() else float("nan"),
                        novel_share_supported=r((s_ & (W[:, 3] > 0.5)).sum(), s_.sum()), novel_share_unsupported=r((u_ & (W[:, 3] > 0.5)).sum(), u_.sum())))
    return out


def potential(split="val", seed=0, ident="e292_a", n=None, cache=LC.CACHE, dir=None, log=print):
    """step 0 of the E368 card (what could a past give?) on the cache of B' seed `seed`: the frames that have a past (the previous keyframe's delivered tokens, p >= 0.5, warped into the lane frame of t by the stored pose; the identity pose and a random frame of
    the stream as nulls, default_rng(0), one draw per frame with a past); n: that many evenly spaced frames (default all) -> dict (JSON-able): GT nodes found / missed by a delivered token, of the missed the share with a weak token near / without, the weak and the
    delivered tokens true / false and supported, the table by existence bin (n, true share, supported when true / false, novel share among the supported / unsupported tokens), the weak tokens by role (novel / duplicate / false) and their support, the recall ceiling"""
    import lane_tok_data as LT
    d = dir or LP.cache_dir(split, seed, ident, None, cache)
    ext, c = load_ext(d), LC.TokCache(d)
    ds = LT.TokData(split, res=2.0, feats=False)
    assert ds.tokens == c.tokens, "the GT tables are not in the order of the cache"
    idx = np.arange(len(c)) if n is None else np.linspace(0, len(c) - 1, n).astype(int)
    rng, acc, rows, with_past = np.random.default_rng(0), {}, [], 0
    arr = lambda i, k: np.array(c.f[k][i], float) if k != "valid" else np.array(c.f[k][i])  # noqa: E731
    for i in idx:
        if ext["prev"][i] < 0:
            continue
        ip, t = int(ext["prev"][i]), ds.table(int(i))
        with_past += 1
        pos, dirv, valid, ep = (arr(i, k) for k in ("pos", "dirv", "valid", "e_p"))
        m = arr(ip, "valid") & (arr(ip, "e_p") >= 0.5)
        pp, pd = arr(ip, "pos")[m], arr(ip, "dirv")[m]
        R, tv = (a_.astype(float) for a_ in pose_of(ext, i))
        j = int(rng.integers(len(c)))
        rm = arr(j, "valid") & (arr(j, "e_p") >= 0.5)
        sup = dict(pose=(pp @ R.T + tv, pd @ R.T), identity=(pp, pd), random=(arr(j, "pos")[rm], arr(j, "dirv")[rm]))
        fc = frame_counts(pos, dirv, valid, ep, t["pos"], t["dirv"], t["flags"], sup)
        rows.append(fc.pop("rows"))
        for k, v in fc.items():
            acc[k] = acc.get(k, 0) + v
    W = np.concatenate(rows) if rows else np.zeros((0, 4))
    g, miss = max(acc.get("gt", 0), 1), max(acc.get("missed", 0), 1)
    r = lambda a, b: float(a) / max(float(b), 1.0)  # noqa: E731
    res = dict(split=split, seed=seed, ident=ident, cache=os.path.relpath(d, ROOT), frames=len(idx), frames_with_past=with_past, frames_without_past=len(idx) - with_past, gt_nodes=acc.get("gt", 0),
               found=r(acc.get("found", 0), g), missed=r(acc.get("missed", 0), g), missed_weak_near=r(acc.get("missed_weak", 0), miss), missed_none=r(acc.get("missed_absent", 0), miss), missed_delivered_past_near=r(acc.get("missed_past", 0), miss),
               missed_weak_and_past=r(acc.get("missed_weak_past", 0), miss),
               weak=dict(true=acc.get("tw", 0), true_supported=r(acc.get("tw_s", 0), acc.get("tw", 0)), false=acc.get("fw", 0), false_supported=r(acc.get("fw_s", 0), acc.get("fw", 0))),
               delivered=dict(true=acc.get("td", 0), true_supported=r(acc.get("td_s", 0), acc.get("td", 0)), false=acc.get("fd", 0), false_supported=r(acc.get("fd_s", 0), acc.get("fd", 0))),
               roles={k: dict(n=acc.get(f"{k}_n", 0), pose=r(acc.get(f"{k}_pose", 0), acc.get(f"{k}_n", 0)), identity=r(acc.get(f"{k}_identity", 0), acc.get(f"{k}_n", 0)), random=r(acc.get(f"{k}_random", 0), acc.get(f"{k}_n", 0)))
                      for k in ("novel", "dup", "false")},
               duplicates_of_the_true_weak=r(acc.get("dup_n", 0), acc.get("novel_n", 0) + acc.get("dup_n", 0)),
               recall_ceiling=dict(all=r(acc.get("missed_weak_past", 0), g), topology=r(acc.get("missed_weak_past_topo", 0), g), start_end=r(acc.get("missed_weak_past_startend", 0), g)),
               topology_share_of_gt=r(acc.get("gt_topo", 0), g), topology_share_of_missed=r(acc.get("missed_topo", 0), miss), start_end_share_of_missed=r(acc.get("missed_startend", 0), miss))
    res["by_bin"] = bin_table(W)
    log(f"potential {split} seed {seed}: {len(idx)} frames, {with_past} with a past; GT nodes {res['gt_nodes']} found {res['found']:.4f} missed {res['missed']:.4f}; "
        f"novel {res['roles']['novel']['n']} supported {res['roles']['novel']['pose']:.3f} (identity {res['roles']['novel']['identity']:.3f}, random {res['roles']['novel']['random']:.3f}); "
        f"false {res['roles']['false']['n']} supported {res['roles']['false']['pose']:.3f}; recall ceiling +{res['recall_ceiling']['all']:.4f} (topology +{res['recall_ceiling']['topology']:.4f})")
    return res


# ------------------------------------------------------------------------------------------------------------------------------ selftest
def refuses(fn, text):
    """fn must fail with an AssertionError whose message holds `text` (a call that is accepted raises a RuntimeError, which no test catches)"""
    try:
        fn()
    except AssertionError as e:
        assert text in str(e), (text, str(e))
        return
    raise RuntimeError(f"accepted although it must be refused ({text})")


def toy_world():
    """two poses of the lane frame in a world (rotation 20 / 35 degrees, translations) and a curved lane seen from both: -> (Rp, tp, Rc, tc, P0, P1) with P0 / P1 the SAME world points in the previous / current lane frame (the points of P1 are the ones P0 maps to)"""
    def rot(a):
        a = np.radians(a)
        return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    Rp, tp, Rc, tc = rot(20.0), np.array([10.0, -4.0]), rot(35.0), np.array([12.5, -1.5])
    s = np.linspace(0, 30, 120)
    W = np.stack([2.0 + 0.5 * s + 0.01 * s ** 2, -3.0 + 0.2 * s + 3.0 * np.sin(s / 9.0)], 1) @ rot(5.0).T + tp               # world points
    return Rp, tp, Rc, tc, (W - tp) @ Rp, (W - tc) @ Rc                                                                       # lane frame = R^T (w - t): p_global = t + R p_lane


def selftest():
    import lane_seq_data as LS
    # the convention: the world points of t-1 in the lane frame of t = R p_prev + t (rel_pose), every wrong convention fails the point test, the headings follow the rotation only
    Rp, tp, Rc, tc, P0, P1 = toy_world()
    R, t = LS.rel_pose(Rc, tc, Rp, tp)
    assert np.allclose(warp_pts(P0, R, t, "warp"), P1, atol=1e-4), "p_cur = R p_prev + t"
    T0, T1 = tangents(P0), tangents(P1)
    for how in ("warp", "identity", "inverse", "transposed"):
        dd, aa = pair_errors(P0, T0, P1, T1, R, t, how)
        s = stats(dd)
        assert point_pass(s) == (how == "warp"), (how, s)
        if how == "warp":
            assert s["n"] > 100 and s["p99"] < 1e-3 and stats(aa)["p90"] < 0.5, (s, stats(aa))
    assert np.allclose(warp_dirs(T0, R, "warp"), T1, atol=1e-3) and np.allclose(warp_dirs(T0, R, "identity"), T0) and not np.allclose(warp_dirs(T0, R, "inverse"), T1, atol=1e-2)
    P = np.array([[1.0, 0.0]])
    R90 = np.array([[0.0, -1.0], [1.0, 0.0]])
    assert np.allclose(warp_pts(P, R90, [2.0, 3.0], "warp"), [[2.0, 4.0]]) and np.allclose(warp_pts(P, R90, [2.0, 3.0], "inverse"), [[-3.0, 1.0]]) and np.allclose(warp_pts(P, R90, [2.0, 3.0], "transposed"), [[2.0, 2.0]]) and np.allclose(warp_pts(P, R90, [2.0, 3.0], "identity"), P)
    assert point_pass(dict(n=5, p90=0.10, p99=0.25)) and not point_pass(dict(n=5, p90=0.11, p99=0.2)) and not point_pass(dict(n=5, p90=0.05, p99=0.26)) and not point_pass(stats([])), "the tolerances are inclusive and an empty set fails"
    d_, a_ = pair_errors(P0, T0, P1 + [0.0, 50.0], T1, R, t, "warp")
    assert len(d_) == 0, "points of t more than 6 m away are not counted"
    up = np.tile([0.0, 1.0], (2, 1))
    ii, jj = pair_errors(np.array([[100.0, 100.0], [1.0, 1.0]]), up, np.array([[100.0, 100.5], [1.0, 1.5]]), up, np.eye(2), np.zeros(2), "warp")
    assert ii.tolist() == [0.5] and jj.tolist() == [0.0], "warped points outside the 45 m box are not counted (only the point at (1, 1))"
    assert token_errors(np.array([[1.0, 1.0], [100.0, 100.0], [20.0, 20.0]]), np.array([[1.0, 2.0], [100.0, 100.5]]), np.eye(2), np.zeros(2), "identity").tolist() == [1.0], "token distances: outside the box and farther than 6 m are not counted"
    good = dict(n=100, median=0.01, p90=0.05, p99=0.2)
    bad = dict(n=100, median=0.3, p90=1.9, p99=5.0)
    pts_ = dict(warp=good, identity=bad, inverse=bad, transposed=bad, shifted=bad)
    hd_, tk_ = dict(warp=dict(n=9, p90=4.9)), dict(warp=dict(n=9, p90=1.5))
    assert all(fixture_checks(pts_, hd_, tk_).values()) and len(fixture_checks(pts_, hd_, tk_)) == 7, "the fixture rule on good numbers"
    for name, (p_, h_, k_) in dict(points=(dict(pts_, warp=dict(good, p99=0.26)), hd_, tk_), headings=(pts_, dict(warp=dict(n=9, p90=5.1)), tk_), tokens=(pts_, hd_, dict(warp=dict(n=9, p90=1.6))),
                                   identity=(dict(pts_, identity=good), hd_, tk_), shifted=(dict(pts_, shifted=good), hd_, tk_), empty=(pts_, dict(warp=dict(n=0, p90=float("nan"))), tk_)).items():
        assert not all(fixture_checks(p_, h_, k_).values()), f"the fixture accepted bad {name}"
    # link: only the rows of the cache are read (the other rows of the poses file are NaN: the TEST rows), the previous row is a cache row, a hand-made pose pair
    n = 12
    pose = dict(prev=-np.ones(n, np.int64), R=np.full((n, 2, 2), np.nan), t=np.full((n, 2), np.nan), scene=np.zeros(n, np.int64), ts=np.arange(n, dtype=float))
    rows = [3, 4, 5, 8, 9]
    pose["prev"][[4, 5, 9]] = [3, 4, 8]
    pose["scene"][[8, 9]] = 1
    for r_ in rows:
        pose["R"][r_], pose["t"][r_] = np.eye(2), 0.0
    pose["R"][4], pose["t"][4] = R90, [5.0, 1.0]                                                       # frame 4: its lane frame is rotated by 90 degrees and moved to (5, 1) in the world; frame 3 is the identity at the origin
    pose["R"][5], pose["t"][5] = np.eye(2), [5.0, 4.0]
    prev, Rl, tl = link(pose, rows)
    assert prev.tolist() == [-1, 0, 1, -1, 3], prev
    assert np.allclose(Rl[1], R90.T) and np.allclose(tl[1], R90.T @ (np.array([0.0, 0.0]) - [5.0, 1.0])), (Rl[1], tl[1])     # Rc^T Rp, Rc^T (tp - tc) with c = frame 4, p = frame 3
    assert np.allclose(Rl[2], R90) and np.allclose(tl[2], [0.0, -3.0]), (Rl[2], tl[2])                  # c = frame 5 (identity at (5, 4)), p = frame 4 (R90 at (5, 1)): R = Rc^T Rp = R90, t = tp - tc = (0, -3)
    assert np.allclose(Rl[0], np.eye(2)) and np.allclose(tl[0], 0.0) and np.allclose(Rl[3], np.eye(2)), "no previous keyframe: identity / zero"
    assert Rl.dtype == np.float32 and tl.dtype == np.float32 and prev.dtype == np.int32
    bad = dict(pose, prev=pose["prev"].copy())
    bad["prev"][5] = 7
    refuses(lambda: link(bad, rows), "not in the cache")
    so = dict(pose)
    check_stream(so, rows, prev, None)
    refuses(lambda: check_stream(so, rows, np.array([-1, 0, 0, -1, 3], np.int32), None), "row right before")
    refuses(lambda: check_stream(so, rows, np.array([-1, 0, 1, -1, -1], np.int32), None), "scene starts of the stream order")
    refuses(lambda: check_stream(so, [4, 3, 5, 8, 9], prev, None), "not the stream order")
    # the TEST guard: VAL rows are exactly the split's rows, in order; a TEST row, a missing row or another order fails
    sv = [39, 40, 41]
    check_rows("val", [39, 40, 41], sv, None)
    for wrong in ([39, 40], [39, 40, 41, 900], [40, 39, 41], [39, 40, 900]):
        refuses(lambda: check_rows("val", wrong, sv, None), "splits.json")
    refuses(lambda: check_rows("train", [1, 2, 2], None, None), "a row twice")
    # the sequences: chains never cross a scene start; the scene ids
    pv = np.array([-1, 0, 1, 2, -1, 4, 5, -1, -1, 8], np.int32)
    assert chain_starts(pv, 3) == [0, 1, 4] and chain_starts(pv, 2) == [0, 1, 2, 4, 5, 8] and chain_starts(pv, 1) == list(range(10)) and chain_starts(pv, 5) == []
    assert scene_ids(pv).tolist() == [0, 0, 0, 0, 1, 1, 1, 2, 3, 3]
    pw = np.array([-1, 0, 1, 2, -1, 4, 5], np.int32)                                                      # the pair choice: the stored previous frame, or the one before it (the shifted mutant)
    assert sorted(pairs(pw, 99)) == [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6)] and sorted(pairs(pw, 99, shift=1)) == [(0, 2), (1, 3), (4, 6)] and len(pairs(pw, 2)) == 2 and pairs(pw, 3, seed=1) == pairs(pw, 3, seed=1)
    ext_ = dict(rel_R=np.stack([np.eye(2) * (i + 1) for i in range(4)]), rel_t=np.arange(8.0).reshape(4, 2))                  # distinct entries: frame i has ITS entry (not its previous frame's)
    R_, t_ = pose_of(ext_, 2)
    assert np.array_equal(R_, np.eye(2) * 3) and np.array_equal(t_, [4.0, 5.0]), "pose_of reads the entry of the frame itself"
    names = []
    _atomic(os.path.join(tempfile.gettempdir(), "past.json"), lambda tmp: (names.append(os.path.basename(tmp)), open(tmp, "w").close()))
    _atomic(os.path.join(tempfile.gettempdir(), "prev.npy"), lambda tmp: (names.append(os.path.basename(tmp)), np.save(tmp, np.zeros(1))))
    assert names == [f"past.{os.getpid()}.tmp.json", f"prev.{os.getpid()}.tmp.npy"], names
    os.remove(os.path.join(tempfile.gettempdir(), "past.json"))
    os.remove(os.path.join(tempfile.gettempdir(), "prev.npy"))
    toy_cache_test()
    frame_counts_test()
    print("lane_past_data selftest ok")


def toy_cache_test():
    """a tiny cache written here: build (idempotent, atomic, the guard files), the loader's items and batches, the stale-cache refusal"""
    F, K = 9, 8
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, "bX_0_K8_sig_v1", "train")
        os.makedirs(d)
        arr = dict(tok=rng.random((F, K, 4)).astype(np.float16), pos=rng.random((F, K, 2)).astype(np.float32), dirv=rng.random((F, K, 2)).astype(np.float32), e=rng.random((F, K)).astype(np.float32), e_raw=rng.random((F, K)).astype(np.float32),
                   s=rng.random((F, K)).astype(np.float32), uout=rng.random((F, K)).astype(np.float32), uin=rng.random((F, K)).astype(np.float32), flags=rng.random((F, K, 8)).astype(np.float32), valid=np.ones((F, K), bool),
                   gt_type=np.zeros((F, K), np.int8), gt_out=np.zeros(F, np.int32), ypair=-np.ones((F, 4, 2), np.int16), ypair_long=np.zeros((F, 4), bool), ypair_scale=np.full((F, 4, 2), np.nan, np.float32),
                   e_p=rng.random((F, K)).astype(np.float32), flags_p=rng.random((F, K, 8)).astype(np.float32), s_p=rng.random((F, K)).astype(np.float32), cell=np.tile(np.arange(K, dtype=np.int16), (F, 1)),
                   gt_id=-np.ones((F, K), np.int32), gd=np.full((F, K), np.nan, np.float32), gt_flags=np.zeros((F, K, 8), bool), gt_aux=np.zeros((F, K, 3), bool))
        for k, v in arr.items():
            np.save(os.path.join(d, k + ".npy"), v)
        rows = [100, 101, 102, 103, 104, 110, 111, 112, 113]                                                # two scenes: rows 100 - 104 and 110 - 113
        json.dump(dict(rows=rows, tokens=[f"t{r}" for r in rows]), open(os.path.join(d, "frames.json"), "w"))
        json.dump(dict(done=F), open(os.path.join(d, "progress.json"), "w"))
        json.dump(dict(split="train"), open(os.path.join(d, "meta.json"), "w"))
        json.dump(dict(frames=F), open(os.path.join(d, "manifest.json"), "w"))
        n = 120
        pose = dict(prev=-np.ones(n, np.int64), R=np.tile(np.eye(2), (n, 1, 1)), t=np.zeros((n, 2)), scene=np.zeros(n, np.int64), ts=np.arange(n, dtype=float))
        for r_ in rows:
            pose["prev"][r_] = r_ - 1 if r_ not in (100, 110) else -1
        pose["scene"][110:] = 1
        pose["t"][:, 1] = 0.25 * np.arange(n) ** 2                                                             # the ego accelerates along +y of the world (every step differs); the lane frame = the world frame except at row 103 (turned by 90 degrees)
        pose["R"][103] = [[0.0, -1.0], [1.0, 0.0]]
        pf = os.path.join(tmp, "poses_train.npz")
        np.savez(pf, **pose)
        dv = os.path.join(tmp, "bX_0_K8_sig_v1", "val")                                                   # the TEST guard of `build` (check=True): a VAL cache that is not data/splits.json['val'] is refused before anything is written
        shutil.copytree(d, dv)
        json.dump(dict(split="val"), open(os.path.join(dv, "meta.json"), "w"))
        refuses(lambda: build(dv, pf, check=True), "splits.json")
        assert not os.path.exists(os.path.join(dv, "past.json")) and not os.path.exists(os.path.join(dv, "prev.npy")), "a refused build wrote files"
        info = build(d, pf, check=False)
        assert info["with_past"] == 7 and info["without_past"] == 2 and info["frames"] == F and info["poses"] == "poses_train.npz" and info["convention"] == CONVENTION, info
        files = {n_: open(os.path.join(d, n_), "rb").read() for n_ in ("prev.npy", "rel_R.npy", "rel_t.npy", "past.json")}
        build(d, pf, check=False)
        assert all(open(os.path.join(d, n_), "rb").read() == b for n_, b in files.items()), "build is not idempotent"
        assert not [f for f in os.listdir(d) if ".tmp" in f], "a temporary file is left"
        ext = load_ext(d)
        assert ext["prev"].tolist() == [-1, 0, 1, 2, 3, -1, 5, 6, 7] and np.allclose(ext["rel_t"][1], [0.0, -50.25]) and np.allclose(ext["rel_t"][2], [0.0, -50.75]), (ext["prev"], ext["rel_t"][:3])        # p_cur = p_prev + (0, 0.25 (r - 1)^2 - 0.25 r^2)
        assert np.allclose(ext["rel_R"][3], [[0.0, 1.0], [-1.0, 0.0]]) and np.allclose(ext["rel_t"][3], [-51.25, 0.0]) and np.allclose(ext["rel_R"][4], [[0.0, -1.0], [1.0, 0.0]]) and np.allclose(ext["rel_t"][4], [0.0, -51.75])      # row 103: Rc^T Rp = R90^T, Rc^T (tp - tc) = R90^T (0, -51.25); row 104 (p = 103): R90, (0, -51.75)
        assert np.allclose(ext["rel_t"][[0, 5]], 0.0) and np.allclose(ext["rel_R"][[0, 5]], np.eye(2)), "identity / zero where there is no previous keyframe"
        ds = PastData("train", dir=d, K_seq=3)
        assert len(ds) == 3 + 2 and [it for it in ds.items] == [[0, 1, 2], [1, 2, 3], [2, 3, 4], [5, 6, 7], [6, 7, 8]], ds.items
        it = ds[1]
        assert len(it["frames"]) == 3 and it["idx"].tolist() == [1, 2, 3] and it["has_prev"].tolist() == [True, True, True] and ds[0]["has_prev"].tolist() == [False, True, True] and it["rel_t"].shape == (3, 2)
        assert it["frames"][2]["row"] == 103 and it["frames"][0]["token"] == "t101" and np.allclose(it["rel_t"], ext["rel_t"][[1, 2, 3]]) and np.allclose(it["rel_R"], ext["rel_R"][[1, 2, 3]])
        b = collate([ds[1], ds[3]])
        assert len(b["frames"]) == 3 and b["rel_R"].shape == (2, 3, 2, 2) and b["rel_t"].shape == (2, 3, 2) and b["has_prev"].shape == (2, 3) and b["has_prev"].dtype == torch.bool and b["idx"].tolist() == [[1, 2, 3], [5, 6, 7]] and np.allclose(b["rel_t"][1].numpy(), ext["rel_t"][[5, 6, 7]]) and np.allclose(b["rel_R"][0].numpy(), ext["rel_R"][[1, 2, 3]])
        assert b["frames"][0]["row"].tolist() == [101, 110] and b["frames"][2]["row"].tolist() == [103, 112] and b["frames"][1]["tok"].shape == (2, K, 4) and b["frames"][2]["y"].shape == (2, K, K)
        sv = PastData("train", dir=d, stream=True)
        assert len(sv) == F and sv.K_seq == 1 and sv[5]["has_prev"].tolist() == [False] and sv[6]["has_prev"].tolist() == [True] and len(sv[0]["frames"]) == 1 and sv.scene.tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1]
        json.dump(dict(rows=rows[::-1], tokens=[f"t{r}" for r in rows[::-1]]), open(os.path.join(d, "frames.json"), "w"))
        refuses(lambda: PastData("train", dir=d), "another cache")
        os.remove(os.path.join(d, "past.json"))
        refuses(lambda: PastData("train", dir=d), "no past.json")


def frame_counts_test():
    """the roles of the weak tokens on a hand-made frame: a lane of GT nodes 2 m apart along y, a delivered token on node 0, weak tokens near node 1 (novel), near node 0 (duplicate) and far from everything (false); the past supports the first and the last"""
    gpos = np.array([[0.0, 0.0], [0.0, 2.0], [0.0, 4.0]])
    gdir = np.tile([0.0, 1.0], (3, 1))
    gfl = np.zeros((3, 8), bool)
    gfl[2, 5] = True                                                                                    # node 2: exit (a topology node)
    h3 = [0.8660254037844386, 0.5]                                                                      # a heading whose cosine with north is exactly 0.5
    pos = np.array([[0.0, 0.1], [0.0, 2.2], [0.1, 0.2], [20.0, 20.0], [0.0, 4.3], [0.0, 0.3], [1.5, 2.0], [0.0, 2.1], [-30.0, -30.0], [0.0, 2.0], [-20.0, -20.0]])
    dirv = np.tile([0.0, 1.0], (11, 1))
    dirv[5] = [0.0, -1.0]                                                                               # a weak token with the opposite heading: near nothing
    dirv[7] = h3
    dirv[10] = [0.0, -1.0]                                                                              # a far weak token against a past token 0.5 m away with the opposite heading: not supported
    valid = np.array([True] * 9 + [False, True])                                                              # token 9 (a padding row near node 1) counts nowhere
    ep = np.array([0.9, 0.3, 0.2, 0.1, 0.45, 0.15, 0.25, 0.35, 0.5, 0.2, 0.4])                              # token 8: exactly 0.5 = delivered; token 6 lies exactly 1.5 m from node 1, token 7 has the cosine 0.5 with it: both near
    sup = dict(pose=(np.array([[0.0, 2.1], [20.0, 19.5], [3.0, 2.0], [-20.0, -19.5]]), np.tile([0.0, 1.0], (4, 1))), identity=(np.zeros((0, 2)), np.zeros((0, 2))), random=(np.array([[0.0, 4.2]]), np.array([[0.0, 1.0]])))
    fc = frame_counts(pos, dirv, valid, ep, gpos, gdir, gfl, sup)
    assert fc["gt"] == 3 and fc["found"] == 1 and fc["missed"] == 2 and fc["missed_weak"] == 2 and fc["missed_absent"] == 0 and fc["missed_past"] == 1 and fc["missed_weak_past"] == 1 and fc["missed_weak_past_topo"] == 0 and fc["gt_topo"] == 1
    assert fc["novel_n"] == 4 and fc["dup_n"] == 1 and fc["false_n"] == 3 and fc["novel_pose"] == 3 and fc["novel_random"] == 1 and fc["novel_identity"] == 0 and fc["false_pose"] == 1 and fc["dup_pose"] == 0, fc
    assert fc["tw"] == 5 and fc["tw_s"] == 3 and fc["fw"] == 3 and fc["fw_s"] == 1 and fc["td"] == 1 and fc["td_s"] == 0 and fc["fd"] == 1 and fc["fd_s"] == 0, fc
    assert fc["rows"].shape == (8, 4) and fc["rows"][:, 0].tolist() == ep[[1, 2, 3, 4, 5, 6, 7, 10]].tolist() and fc["rows"][:, 1].tolist() == [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    assert fc["rows"][:, 2].tolist() == [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0] and fc["rows"][:, 3].tolist() == [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0], fc["rows"]
    none = frame_counts(pos, dirv, valid, ep, np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 8), bool), sup)
    assert "gt" not in none and none["tw"] == 0 and none["fw"] == 8 and none["fd"] == 2 and none["rows"][:, 3].sum() == 0 and none["rows"][:, 1].sum() == 0, none
    # the table by existence bin, [lo, hi): by hand
    tb = bin_table(fc["rows"])
    assert [b["n"] for b in tb] == [0, 2, 2, 2, 2] and [b["true_share"] for b in tb] == [0.0, 0.0, 1.0, 1.0, 0.5], tb
    assert tb[1]["supported_if_false"] == 0.5 and tb[2]["supported_if_true"] == 0.5 and tb[3]["supported_if_true"] == 1.0 and tb[4]["supported_if_true"] == 0.0 and tb[4]["supported_if_false"] == 0.0 and np.isnan(tb[1]["supported_if_true"]) and np.isnan(tb[0]["supported_if_false"])
    assert [b["novel_share_supported"] for b in tb] == [0.0, 0.0, 1.0, 1.0, 0.0] and [b["novel_share_unsupported"] for b in tb] == [0.0, 0.0, 0.0, 0.0, 0.5], tb
    # the heading angle by hand: the same line with the tangents turned by 30 degrees
    P0 = np.stack([np.zeros(20), 2.0 * np.arange(20.0)], 1)                                              # points 2 m apart: the tangents must be normalised
    Tr = np.tile([np.sin(np.radians(30.0)), np.cos(np.radians(30.0))], (20, 1))
    dd, aa = pair_errors(P0, tangents(P0), P0, Tr, np.eye(2), np.zeros(2), "warp")
    assert len(aa) == 20 and np.allclose(aa, 30.0, atol=1e-6) and np.allclose(dd, 0.0)
    s_ = stats(np.arange(101.0))
    assert s_ == dict(n=101, median=50.0, p90=90.0, p99=99.0), s_


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("build")
    s.add_argument("--cache", help="one cache directory <run>/<split> (default: all eight under runs/e365_cache)")
    s.add_argument("--all", action="store_true")
    for name, n_default in (("fixture", 200),):
        s = sub.add_parser(name)
        s.add_argument("--split", default="val", choices=("val", "train"))
        s.add_argument("--n-pairs", type=int, default=n_default)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--ident", default="e292_a")
        s.add_argument("--out")
    s = sub.add_parser("potential")
    s.add_argument("--split", default="val", choices=("val", "train"))
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--ident", default="e292_a")
    s.add_argument("--n", type=int)
    s.add_argument("--out")
    sub.add_parser("check")
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "check":
        check_built()
    elif a.cmd == "build":
        for d in ([os.path.normpath(a.cache if os.path.isabs(a.cache) else os.path.join(ROOT, a.cache))] if a.cache else cache_dirs()):
            info = build(d)
            print(f"{os.path.relpath(d, ROOT)}: {info['frames']} frames, {info['with_past']} with a past, {info['without_past']} without; poses {info['poses']} ({info['poses_sha']})")
    elif a.cmd == "fixture":
        res = fixture(a.split, a.n_pairs, a.seed, a.ident)
        if a.out:
            _save_json(a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out), res)
        assert res["passed"], f"the {a.cmd} FAILED"
    else:
        res = potential(a.split, a.seed, a.ident, a.n)
        if a.out:
            _save_json(a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out), res)
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
