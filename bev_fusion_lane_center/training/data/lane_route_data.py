"""E389 (A6, D127): the data of the SD-ROUTE association of the champion tokens: the driven route of a frame from the lane GT, the labels of the GT nodes (on the route's road / on the ego lane), the SD route tokens of a navigation map with the
protocol's map errors, a derived cache of all that, and the loader on top of `lane_ptok_data`. The map is a GT-derived SIMULATION: the route is the centred path of the lane pieces that the ego path follows (`sd_lib.build_sd`, the lane choice is lost);
no piece id, road id, lane index or node label ever enters a route token.

    wsl bash tools/env/_py14.sh lane_route_data.py build   --split val|train [--limit N]       # runs/e389_route/<split>/ (derived data, not in git; atomic, idempotent: a finished cache is left as it is)
    wsl bash tools/env/_py14.sh lane_route_data.py check   [--n N] [--split val|train|both]                             # the TEST guards (VAL = splits['val'], TRAIN = splits['train'], no VAL / TEST frame in TRAIN), the leak check and the node <-> piece alignment on built tables, the cache against a recompute
    wsl bash tools/env/_py14.sh lane_route_data.py loadercheck                                  # the defaults of `loader` on the real caches (TRAIN shuffled / drop_last / mode train, VAL in order / mode error)
    wsl bash tools/env/_py14.sh lane_route_data.py fixture --split val [--n 200] [--out FILE]  # the card's data fixture (section 1 a - d): PASS / FAIL + numbers; the wrong conventions must FAIL
    wsl bash tools/env/_py14.sh lane_route_data.py stats   --split val|train [--ident e292_a] [--seed 0] [--out FILE]   # route share, label shares of the covered GT nodes, route tokens, decision frames
    wsl bash tools/env/_py14.sh lane_route_data.py selftest                                    # no cache needed; tests: grid/test_lane_route.py (the hand mutants)
    ds, dl = loader("train" | "val", seed=0, bs=8, workers=4, init_seed=0, mode="auto", ident="e292_a")       # the batches of `lane_ptok_data.loader` + the route and the labels
    batch["route"]    dict(pos (B, M, 2) float32 [lane frame, m], dir (B, M, 2) the unit direction of travel, count (B, M) int64 the lane count 1 - 5 (0 = unknown), valid (B, M) bool); M = the largest route of the batch (at least 1; padding is not valid)
                      absent when mode is None; mode "none" (alias "absent") = no samples (the none key only: arm z)
    batch["on_road"], batch["on_lane"], batch["on_road_id"]   (B, K) int8 in {-1, 0, 1}: the label of the GT node that the token is assigned to (the cache's `gt_id`); -1 = no node, and -1 for every token of a frame without a route.
                      on_road_id is a DIAGNOSTIC (the road-id label, see below), not the label of the association
    batch["has_route"], batch["decision"]   (B,) bool: the frame has a route; a route fork lies within 40 m ahead (the frames where the map matters)

Route of a frame (`route_of`) = the GT's own tier-1 (`GtGenerator/lane_tiers.py`: the driven path of the car, 25 m back to 55 m ahead, extended along the map to 40 m ahead where the car stops or the sequence ends): the pieces with a tier-1 point, in driving
order (`driving_order`); so a stopped ego still has a route and a frame has none only when no piece has a tier-1 point (has_route False, every label -1, no route tokens). `viterbi_route` (the pieces the ego path follows: a Viterbi over piece vertices
within 2.5 m, heading cosine >= cos 45 degrees, as `make_route_gt.route_for`) is only the cross-check that `fixture` reports. Labels per GT node of the E357 table (`lane_tok_data`, `piece` = the index into g['pieces']): on_lane = its piece is a route piece;
on_road = (the node lies within n * 1.75 m + 0.5 m of the EXACT SD route polyline (the polyline the route tokens are made from, before the map error; n = the lane count at the nearest route point) AND its heading cosine to the route direction there is >= 0.7)
or it is on_lane; on_road_id (diagnostic) = its piece belongs to a ROAD that contains a route piece (`g['roads'][i]['pieces']`; a junction road holds ALL its connectors, so it blurs the fork decision) or it is on_lane. decision = FORKS ONLY: a route piece with more than one successor whose end lies within 40 m in
front of the ego along the route (a merge creates no route ambiguity).
Route tokens (`sample_route`): the route polylines (one per route piece, centred over the parallel lanes, a lane count at every vertex) after the map error, sampled every 2 m: pos, unit tangent in the direction of travel, the lane count (interpolated, rounded, 1 - 5);
at most 160 samples, the nearest the ego are kept (`check` counts the truncated frames). FIXED KEY SET (pos, dir, count, valid): `check_keys` refuses any other key. Modes (`RouteData(mode=)`; the map-error draws of the E374 map loader, applied to the route):
  exact   no error                                    error  a FIXED draw per frame (a generator seeded by the frame token, md5): shift 1 - 2 m in a random direction, yaw N(0, 1 degree), `sd_lib.degrade_sd` (Douglas-Peucker 0.5 - 2 m, count errors 0.1)
  train   one DRAW per read (TRAIN): exact 25 %, error-like 47 % (shift U(0, 2) m), counts unknown 8 %, absent 10 %, gross 10 % (50 %: shift U(4, 8) m, 50 %: the error-like route of a frame of ANOTHER scene); the generator is seeded by (the init seed, the frame token,
          the draw index of the frame: a counter array in shared memory, so persistent workers count together); a RESUMED run starts its counters at 0 again
  gross   the `error` route shifted rigidly by 5 m   none  no samples   wrong  the `error` route of ANOTHER scene (frame (i + 97) mod N, moved on to the next frame of another scene that has a route: a fixed plan)
The labels never change with the map error (they come from the GT). Derived cache `runs/e389_route/<split>/`: per frame the labels (padded to the largest table, -1), has_route, decision, the exact route polylines (ragged) and the exact route tokens (padded to 160),
the scene id; manifest.json (version, lane GT signature, counts, sha256 of the frame token list) last; the reader refuses another version, another GT or another frame list.
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import zlib

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub_ in ("training/data", "training/data/prep", "training/grid", "evaluation", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import lane_past_data as LPD  # noqa: E402
import lane_ptok_data as LP  # noqa: E402
import lane_tokcache as LC  # noqa: E402
import sd_lib as SD  # noqa: E402

VERSION = "v4"
CACHE = os.path.join(ROOT, "runs", "e389_route")
KEYS = ("pos", "dir", "count", "valid")                                      # the fixed key set of a route table
MAX_SAMPLES = 160                                                            # route samples per frame: the nearest the ego are kept (the raw route is at most 171 samples: 0.00 % of VAL, 0.04 % of TRAIN above 160)
STEP = 2.0                                                                   # m between the samples of a route polyline
COUNT_MAX = 5                                                                # the lane counts of the table are 1 - COUNT_MAX (0 = unknown)
MAX_D, COS_MIN = 2.5, float(np.cos(np.deg2rad(45.0)))                       # the route match: a piece vertex within 2.5 m of a path point, heading cosine >= cos 45 degrees
COST_SUCC, COST_JUMP = 0.3, 3.0                                              # the Viterbi change to a successor piece / to another piece (`make_route_gt.route_for`)
BAND_LANE, BAND_BASE, BAND_COS = 1.75, 0.5, 0.7                         # on_road: within n * 1.75 + 0.5 m of the exact route polyline, heading cosine >= 0.7
DUP_D, DUP_COS = 1.0, 0.9                                                    # route polylines: a vertex within 1 m of an earlier one with a heading cosine >= 0.9 is the same road (the pieces of a fork share their first metres)
AHEAD = 40.0                                                                 # m along the driven path: a fork / merge within it makes a decision frame
ERROR_SHIFT, TRAIN_SHIFT, GROSS_SHIFT = (1.0, 2.0), (0.0, 2.0), (4.0, 8.0)   # m: the shift of the `error` route, of the error-like draws of `train`, of the gross draws
GROSS_M = 5.0                                                                # mode `gross`: the `error` route shifted rigidly by this (m)
OTHER = 97                                                                   # the plan of `wrong`: frame (i + OTHER) mod N, moved on to another scene
MIX = (("exact", 0.25), ("error", 0.47), ("unknown", 0.08), ("absent", 0.10), ("gross", 0.10))     # the draw classes of mode `train`
MODES = ("exact", "error", "train", "gross", "none", "wrong")
EXPECT = dict(val=873, train=8414)
EMPTY = (np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0))                    # a route without samples (pos, dir, count)
ARRAYS = dict(n_nodes="int32", on_road="int8", on_lane="int8", on_road_id="int8", has_route="bool", decision="bool", n_route_pieces="int16", scene="int32", poly_xy="float64", poly_cnt="float64", poly_end="int64", frame_poly_end="int64",
              rt_pos="float32", rt_dir="float32", rt_count="int8", rt_n="int16", rt_cut="int16")


# ------------------------------------------------------------------------------------------------------------------------------ the route of a frame (GT)
def densify(path, step=1.0):
    """a polyline (n, 2) -> points every <= `step` m along it (the vertices of the polyline are kept)"""
    path = np.asarray(path, float)
    out = [path[:1]]
    for a, b in zip(path[:-1], path[1:]):
        n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
        out.append(a + (b - a) * (np.arange(1, n + 1) / n)[:, None])
    return np.concatenate(out)


def unit_tangents(p):
    """unit direction of travel at every point of a polyline (central differences; zeros for a polyline of one point)"""
    if len(p) < 2:
        return np.zeros((len(p), 2))
    t = np.gradient(p, axis=0)
    return t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)


def piece_vertices(g):
    """the vertices of the pieces with >= 2 points: (V (n, 2), piece index (n,), unit tangent (n, 2))"""
    V, pid, tan = [], [], []
    for k, p in enumerate(g["pieces"]):
        q = np.asarray(p["pts"], float)
        if len(q) >= 2:
            V.append(q)
            tan.append(unit_tangents(q))
            pid.append(np.full(len(q), k))
    return (np.concatenate(V), np.concatenate(pid), np.concatenate(tan)) if V else (np.zeros((0, 2)), np.zeros(0, int), np.zeros((0, 2)))


def successors(g):
    """the set of (piece index, successor piece index) of the frame (a successor outside the frame is left out)"""
    idx = {p["id"]: k for k, p in enumerate(g["pieces"])}
    return {(k, idx[s]) for k, p in enumerate(g["pieces"]) for s in p["succ"] if s in idx}


def viterbi_route(g, path=None, max_d=MAX_D, cos_min=COS_MIN):
    """the CROSS-CHECK of the route (not the label): the pieces that the ego path follows, in driving order: the driven path (`g['ego_path']`, or `path`) densified to 1 m, inside the window, a Viterbi over the pieces (candidate = a vertex within 2.5 m with a heading
    cosine >= cos 45 degrees, cost = the distance, a change to a successor piece 0.3, to another piece 3.0, as `make_route_gt.route_for`); [] = no match. `fixture` reports its agreement with `route_of`"""
    path = np.asarray(g["ego_path"]["pts"] if path is None else path, float)
    V, pid, tan = piece_vertices(g)
    if len(V) == 0 or len(path) < 2:
        return []
    tree, succ, half = cKDTree(V), successors(g), float(g["half"])
    dp = densify(path)
    tp = unit_tangents(dp)
    cands = []
    for q, tq in zip(dp, tp):
        if abs(q[0]) > half or abs(q[1]) > half:
            continue
        cd = {}
        for c in tree.query_ball_point(q, max_d):
            if float(tan[c] @ tq) < cos_min:
                continue
            d, k = float(np.linalg.norm(V[c] - q)), int(pid[c])
            if k not in cd or d < cd[k]:
                cd[k] = d
        if cd:
            cands.append(cd)
    if not cands:
        return []
    prev, back = {}, []
    for cd in cands:
        cost, bp = {}, {}
        for k, d in cd.items():
            if not prev:
                cost[k], bp[k] = d, None
                continue
            best = min((c + (0.0 if kk == k else (COST_SUCC if (kk, k) in succ else COST_JUMP)) + d, kk) for kk, c in prev.items())
            cost[k], bp[k] = best
        back.append(bp)
        prev = cost
    k = min(prev, key=prev.get)
    seq = [k]
    for bp in reversed(back[1:]):
        k = bp[k]
        seq.append(k)
    out = []
    for k in reversed(seq):
        if k not in out:
            out.append(k)
    return out


def driving_order(g, ks):
    """the pieces `ks` in driving order: a topological order by the successor links among them; among the pieces that nothing in `ks` precedes, the one whose first tier-1 point lies nearest the start of the driven path comes first (ties: the piece index); a cycle is
    broken by the same key"""
    path = np.asarray(g["ego_path"]["pts"], float)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))] if len(path) else np.zeros(1)
    tree = cKDTree(path) if len(path) else None
    key = {}
    for k in ks:
        pts, tier = np.asarray(g["pieces"][k]["pts"], float), np.asarray(g["pieces"][k]["tier"])
        key[k] = float(arc[tree.query(pts[int(np.argmax(tier == 1))])[1]]) if tree is not None else 0.0
    idx = {p["id"]: k for k, p in enumerate(g["pieces"])}
    left, out = set(ks), []
    while left:
        free = [k for k in left if not any(idx.get(s) == k for j in left if j != k for s in g["pieces"][j]["succ"])] or list(left)
        k = min(free, key=lambda k_: (key[k_], k_))
        out.append(k)
        left.discard(k)
    return out


def route_of(g):
    """the route of a frame = the GT's own tier-1: the pieces with a tier-1 point (the driven path, extended along the map to `min_ahead` 40 m ahead: `GtGenerator/lane_tiers.py`), in driving order; [] when no piece has one"""
    return driving_order(g, [k for k, p in enumerate(g["pieces"]) if (np.asarray(p["tier"]) == 1).any()])


def road_pieces(g, route):
    """the ids of the pieces that belong to a ROAD that contains a route piece (`g['roads'][i]['pieces']`: a road merges the parallel lanes of a carriageway; a piece belongs to every road that runs along it)"""
    ids = {g["pieces"][k]["id"] for k in route}
    return {p for r in g["roads"] if ids & set(r["pieces"]) for p in r["pieces"]}


def route_curve(polys, step=0.25):
    """the exact route polylines [(poly, count)] densified to `step` m: (cKDTree over the points, unit tangents in the direction of travel, the lane count at every point: interpolated, rounded, clipped to 1 - COUNT_MAX); None without a polyline"""
    P, T, C = [], [], []
    for poly, cnt in polys:
        poly = np.asarray(poly, float)
        s_ = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))] if len(poly) > 1 else np.zeros(1)
        if len(poly) < 2 or s_[-1] < 1e-6:
            continue
        t = np.linspace(0.0, s_[-1], max(int(np.ceil(s_[-1] / step)), 1) + 1)
        q = np.stack([np.interp(t, s_, poly[:, 0]), np.interp(t, s_, poly[:, 1])], 1)
        P.append(q)
        T.append(unit_tangents(q))
        C.append(np.clip(np.rint(np.interp(t, s_, np.asarray(cnt, float))), 1, COUNT_MAX))
    return (cKDTree(np.concatenate(P)), np.concatenate(T), np.concatenate(C)) if P else None


def road_band(polys, pos, dirv):
    """(M,) bool: the nodes (pos (M, 2), heading dirv (M, 2)) that lie within BAND_LANE * n + BAND_BASE m of the nearest point of the exact route polylines (n = the lane count there) with a heading cosine >= BAND_COS to the route direction there (the carriageway of the route)"""
    cv = route_curve(polys)
    pos = np.asarray(pos, float).reshape(-1, 2)
    if cv is None or len(pos) == 0:
        return np.zeros(len(pos), bool)
    d, j = cv[0].query(pos)
    return (d <= BAND_LANE * cv[2][j] + BAND_BASE) & ((np.asarray(dirv, float).reshape(-1, 2) * cv[1][j]).sum(1) >= BAND_COS)


def node_labels(g, route, table, polys):
    """the labels of the GT nodes of the E357 table (`table['piece']` = the piece index, `table['pos']` / `table['dirv']`) -> (on_road, on_lane, on_road_id) (M,) int8: on_lane = the node's piece is a route piece; on_road = `road_band` on the exact route polylines `polys`
    (`route_polys`) or on_lane; on_road_id (diagnostic) = its piece belongs to a road that contains a route piece, or on_lane"""
    piece = np.asarray(table["piece"], int)
    assert len(piece) == 0 or (piece.min() >= 0 and piece.max() < len(g["pieces"])), "a node of the table has no piece of the frame"
    lane_ok = np.zeros(len(g["pieces"]), bool)
    lane_ok[list(route)] = True
    roads = road_pieces(g, route)
    id_ok = np.array([p["id"] in roads for p in g["pieces"]], bool) | lane_ok
    on_lane = lane_ok[piece]
    return (road_band(polys, table["pos"], table["dirv"]) | on_lane).astype(np.int8), on_lane.astype(np.int8), id_ok[piece].astype(np.int8)


def decision_frame(g, route, ahead=AHEAD):
    """FORKS ONLY (a merge creates no route ambiguity): a route piece with more than one successor whose end lies within `ahead` m in front of the ego, measured along the route (the pieces in driving order, their lengths added; the ego = the route vertex nearest to the
    keyframe position (0, 0))"""
    if not route:
        return False
    pts = [np.asarray(g["pieces"][k]["pts"], float) for k in route]
    along = [np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))] for p in pts]
    base = np.r_[0.0, np.cumsum([a[-1] for a in along])]
    V, off = np.concatenate(pts), np.concatenate([base[j] + a for j, a in enumerate(along)])
    s_ego = off[int(np.argmin(np.linalg.norm(V, axis=1)))]
    return any(len(g["pieces"][k]["succ"]) > 1 and 0.0 <= base[j + 1] - s_ego <= ahead for j, k in enumerate(route))


def to_curve(polys, step=0.25):
    """the polylines [(n, 2)] densified to `step` m as (cKDTree over the points, unit tangents in the direction of travel); None without a polyline"""
    D = [densify(p, step) for p in polys if len(p) >= 2]
    return (cKDTree(np.concatenate(D)), np.concatenate([unit_tangents(d) for d in D])) if D else None


def curve_dist(curve, pts, dirs=None, cos=0.5):
    """the distance of the points to the nearest curve point; with `dirs` (their directions of travel) a point whose nearest curve point runs against it (heading cosine < `cos`) counts as infinitely far; no curve: infinitely far"""
    if curve is None:
        return np.full(len(pts), np.inf)
    d, j = curve[0].query(pts)
    return d if dirs is None else np.where((dirs * curve[1][j]).sum(1) >= cos, d, np.inf)


def dedupe_polys(polys, tol=DUP_D, cos=DUP_COS):
    """the route polylines [(poly, count)] without the stretches that an earlier polyline already covers (the pieces of a fork or a junction share their first metres): a vertex within `tol` m of an earlier polyline with a heading cosine >= `cos` is cut out,
    the runs of >= 2 vertices that are left are kept (in order)"""
    out, curve = [], None
    for poly, cnt in polys:
        poly, cnt = np.asarray(poly, float), np.asarray(cnt, float)
        if len(poly) < 2:
            continue
        covered = np.zeros(len(poly), bool)
        if curve is not None:
            d, j = curve[0].query(poly)
            covered = (d <= tol) & ((unit_tangents(poly) * curve[1][j]).sum(1) >= cos)
        edges = np.flatnonzero(np.diff(np.concatenate([[0], (~covered).astype(np.int8), [0]])))
        out += [(poly[x:y], cnt[x:y]) for x, y in zip(edges[::2], edges[1::2]) if y - x >= 2]
        curve = to_curve([q for q, _ in out])
    return out


def route_polys(g, route):
    """the SD route of the frame: one (polyline (n, 2), lane count (n,)) per route piece in driving order (`sd_lib.build_sd`: the centred path of the piece with its parallel same-direction lanes, resampled every <= 2 m; it reads g['pts'] only), without the
    stretches that an earlier piece already covers (`dedupe_polys`)"""
    return dedupe_polys(SD.build_sd(dict(pts=[np.asarray(p["pts"], float) for p in g["pieces"]]), list(route))["route"])


def frame_record(g, table):
    """everything the cache keeps of one frame: the route, the labels of the nodes of the E357 table (`node_labels`; all -1 without a route), has_route, decision, the exact route polylines"""
    route = route_of(g)
    n = len(table["piece"])
    if not route:
        return dict(route=[], has_route=False, decision=False, on_road=-np.ones(n, np.int8), on_lane=-np.ones(n, np.int8), on_road_id=-np.ones(n, np.int8), polys=[])
    polys = route_polys(g, route)
    on_road, on_lane, on_road_id = node_labels(g, route, table, polys)
    return dict(route=route, has_route=True, decision=decision_frame(g, route), on_road=on_road, on_lane=on_lane, on_road_id=on_road_id, polys=polys)


# ------------------------------------------------------------------------------------------------------------------------------ route tokens and the map errors
def token_rng(token, salt=""):
    """the generator of a frame token (md5, as `map_sim._rng_for` and the E374 map loader; a salt gives another stream of the same token)"""
    return np.random.RandomState(int(hashlib.md5((str(token) + salt).encode()).hexdigest()[:8], 16))


def rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def pose_error(rng, lo, hi):
    """the rigid pose error of a map, the draws of `map_sim.make_map` in its order: the direction, the length U(lo, hi) m, the yaw N(0, 1 degree) -> (shift (2,), yaw rad)"""
    ang = rng.uniform(0, 2 * np.pi)
    shift = rng.uniform(lo, hi) * np.array([np.cos(ang), np.sin(ang)])
    return shift, np.deg2rad(rng.randn())


def sample_route(polys, shift=None, yaw=0.0):
    """the route polylines [(poly (n, 2), count (n,))], after the pose error p -> R(yaw) p + shift when `shift` is given, sampled every STEP m -> (pos (S, 2), dir (S, 2) the unit tangent in the direction of travel, count (S,) the lane count interpolated along the
    vertices and rounded; not yet capped). A polyline of one point is left out; the first sample of a polyline that lies within 0.5 m of the last sample of the one before (the joint of two pieces) is dropped"""
    P, T, C = [], [], []
    for poly, cnt in polys:
        poly = np.asarray(poly, float)
        if len(poly) < 2:
            continue
        if shift is not None:
            poly = poly @ rot(yaw).T + shift
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))])
        if s[-1] < 1e-6:
            continue
        t = np.linspace(0.0, s[-1], max(int(np.ceil(s[-1] / STEP)), 1) + 1)
        q = np.stack([np.interp(t, s, poly[:, 0]), np.interp(t, s, poly[:, 1])], 1)
        tg = np.gradient(q, axis=0)
        tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-9)
        c = np.interp(t, s, np.asarray(cnt, float))
        if P and np.linalg.norm(q[0] - P[-1][-1]) < 0.5:
            q, tg, c = q[1:], tg[1:], c[1:]
        P.append(q)
        T.append(tg)
        C.append(c)
    return (np.concatenate(P), np.concatenate(T), np.rint(np.concatenate(C))) if P else EMPTY


def error_like(polys, rng, lo, hi):
    """a degraded and displaced route: the pose error first (`pose_error`), then `sd_lib.degrade_sd` on the route polylines (the same generator), then the samples"""
    shift, yaw = pose_error(rng, lo, hi)
    return sample_route(SD.degrade_sd(dict(route=polys, roads=[]), rng)["route"], shift, yaw)


def error_route(polys, token):
    """the FIXED `error` route of a frame: shift 1 - 2 m, yaw N(0, 1 degree), degrade_sd, the generator seeded by the token"""
    return error_like(polys, token_rng(token), *ERROR_SHIFT)


def shifted(raw, vec):
    return (raw[0] + vec, raw[1], raw[2])


def with_count(raw, c):
    return (raw[0], raw[1], np.full(len(raw[0]), float(c)))


def truncate(raw, n=MAX_SAMPLES):
    """at most n samples: the n nearest the ego (the origin), in their own order"""
    pos, d, c = raw
    if len(pos) <= n:
        return raw
    keep = np.sort(np.argsort(np.linalg.norm(pos, axis=1), kind="stable")[:n])
    return pos[keep], d[keep], c[keep]


def check_keys(m):
    """THE leak check of a route table: exactly the keys of KEYS (a piece id, road id, route piece list, lane index or node label is refused), one row per sample"""
    extra, missing = sorted(set(m) - set(KEYS)), sorted(set(KEYS) - set(m))
    assert not extra and not missing, f"a route table has exactly the keys {KEYS}: extra {extra} (a lane-level field is a leak), missing {missing}"
    n = len(m["valid"])
    assert tuple(np.shape(m["pos"])) == (n, 2) and tuple(np.shape(m["dir"])) == (n, 2) and tuple(np.shape(m["count"])) == (n,), f"route table shapes {[np.shape(m[k]) for k in KEYS]}"
    return m


def finish(raw):
    """a raw route -> the table of a frame (numpy; `check_keys` holds): capped to MAX_SAMPLES (nearest the ego), the counts clipped to 1 - COUNT_MAX (0 stays 0 = unknown), all samples valid"""
    pos, d, c = truncate(raw)
    c = np.where(c > 0, np.clip(c, 1, COUNT_MAX), 0)
    return check_keys(dict(pos=pos.astype(np.float32), dir=d.astype(np.float32), count=c.astype(np.int64), valid=np.ones(len(pos), bool)))


def collate_routes(tables):
    """the tables of a batch -> dict of tensors padded to the largest route (at least 1 column; the padding is not valid): pos / dir float32, count int64, valid bool"""
    for t in tables:
        check_keys(t)
    M = max(1, max(len(t["valid"]) for t in tables))
    out = dict(pos=np.zeros((len(tables), M, 2), np.float32), dir=np.zeros((len(tables), M, 2), np.float32), count=np.zeros((len(tables), M), np.int64), valid=np.zeros((len(tables), M), bool))
    for b, t in enumerate(tables):
        n = len(t["valid"])
        for k in KEYS:
            out[k][b, :n] = t[k]
    return {k: torch.from_numpy(v) for k, v in out.items()}


def draw_class(u):
    """the class of a `train` draw from u in [0, 1): MIX in order (exact 25 %, error 47 %, unknown 8 %, absent 10 %, gross 10 %)"""
    c = 0.0
    for name, p in MIX:
        c += p
        if u < c - 1e-12:
            return name
    return MIX[-1][0]


def other_frame(scene, ok, i, j):
    """from the frame j on (+1 each), the first frame of another scene than frame i that has a route (`ok` (N,) bool)"""
    n = len(scene)
    for _ in range(n):
        if scene[j % n] != scene[i] and ok[j % n]:
            return j % n
        j += 1
    raise AssertionError("no frame of another scene has a route")


def other_scene_plan(scene, ok=None, shift=OTHER):
    """(N,) int: the frame whose route frame i borrows (`wrong`): (i + OTHER) mod N, moved on frame by frame until it lies in another scene and has a route. A fixed plan"""
    scene = np.asarray(scene)
    assert len(set(scene.tolist())) >= 2, "one scene only: no route of another scene"
    ok = np.ones(len(scene), bool) if ok is None else np.asarray(ok, bool)
    return np.array([other_frame(scene, ok, i, (i + shift) % len(scene)) for i in range(len(scene))], np.int64)


# ------------------------------------------------------------------------------------------------------------------------------ the derived cache
def pack(recs, scene):
    """the records of `frame_record` (one per frame) and the scene id of every frame -> the arrays of ARRAYS (the labels padded to the largest table with -1, the polylines ragged, the exact route tokens padded to MAX_SAMPLES)"""
    F = len(recs)
    n_nodes = np.array([len(r["on_road"]) for r in recs], np.int32)
    out = dict(n_nodes=n_nodes, on_road=-np.ones((F, max(1, int(n_nodes.max()) if F else 1)), np.int8), has_route=np.array([r["has_route"] for r in recs], bool), decision=np.array([r["decision"] for r in recs], bool),
               n_route_pieces=np.array([len(r["route"]) for r in recs], np.int16), scene=np.asarray(scene, np.int32), rt_pos=np.zeros((F, MAX_SAMPLES, 2), np.float32), rt_dir=np.zeros((F, MAX_SAMPLES, 2), np.float32),
               rt_count=np.zeros((F, MAX_SAMPLES), np.int8), rt_n=np.zeros(F, np.int16), rt_cut=np.zeros(F, np.int16))
    out["on_lane"], out["on_road_id"] = out["on_road"].copy(), out["on_road"].copy()
    xy, cnt, poly_end, frame_end = [], [], [], []
    for i, r in enumerate(recs):
        out["on_road"][i, :n_nodes[i]], out["on_lane"][i, :n_nodes[i]], out["on_road_id"][i, :n_nodes[i]] = r["on_road"], r["on_lane"], r["on_road_id"]
        for poly, c in r["polys"]:
            xy.append(np.asarray(poly, float))
            cnt.append(np.asarray(c, float))
            poly_end.append(sum(len(a) for a in xy))
        frame_end.append(len(poly_end))
        raw = sample_route(r["polys"])
        t = finish(raw)
        m = len(t["valid"])
        out["rt_pos"][i, :m], out["rt_dir"][i, :m], out["rt_count"][i, :m], out["rt_n"][i], out["rt_cut"][i] = t["pos"], t["dir"], t["count"], m, max(len(raw[0]) - MAX_SAMPLES, 0)
    out.update(poly_xy=np.concatenate(xy) if xy else np.zeros((0, 2)), poly_cnt=np.concatenate(cnt) if cnt else np.zeros(0), poly_end=np.array(poly_end, np.int64), frame_poly_end=np.array(frame_end, np.int64))
    return out


def tokens_sha(tokens):
    return hashlib.sha256("\n".join(tokens).encode()).hexdigest()


def cache_params():
    """the module constants that a route cache depends on: stored in its manifest, compared by the reader (the band, duplicate and count constants are covered by VERSION)"""
    return dict(max_samples=MAX_SAMPLES, step=STEP, max_d=MAX_D, cos_min=COS_MIN, ahead=AHEAD, cost_succ=COST_SUCC, cost_jump=COST_JUMP)


def token_lane_gt(tok):
    """the lane GT signature that a token cache was built for (its meta.json); None for a stand-in without a directory"""
    d = getattr(tok, "dir", None)
    return json.load(open(os.path.join(d, "meta.json")))["lane_gt"] if d else None


def write_cache(d, arrays, tokens, meta):
    """arrays of `pack` + the frame tokens + the manifest fields `meta` -> the directory `d`: every file written whole and renamed, manifest.json last (the cache is finished when it exists)"""
    os.makedirs(d, exist_ok=True)
    for n, a in arrays.items():
        LPD._save_npy(os.path.join(d, n + ".npy"), np.asarray(a, ARRAYS[n]))
    LPD._save_json(os.path.join(d, "tokens.json"), tokens)
    ok = arrays["has_route"]
    man = dict(meta, version=VERSION, frames=len(tokens), tokens_sha=tokens_sha(tokens), frames_with_route=int(ok.sum()), frames_without_route=int((~ok).sum()), decision_frames=int(arrays["decision"].sum()), route_tokens_mean=float(arrays["rt_n"].mean()),
               route_tokens_max=int(arrays["rt_n"].max()), frames_truncated=int((arrays["rt_cut"] > 0).sum()), max_nodes=int(arrays["n_nodes"].max()),
               params=cache_params())
    LPD._save_json(os.path.join(d, "manifest.json"), man)
    return man


class TokenRoute:
    """read access to a route cache directory (refused: no manifest, another version, other constants, another lane GT than `lane_gt` when given, another frame list): `rc.tokens`, `rc.a` (the arrays by name, memory maps), `rc.manifest`; `labels(i)` = (on_road, on_lane) of the table nodes of frame i, `labels_id(i)` = the diagnostic on_road_id, `polys(i)` the exact route polylines, `exact(i)` the exact route table (`finish`)"""

    def __init__(self, d, lane_gt=None):
        self.dir = d
        mp = os.path.join(d, "manifest.json")
        assert os.path.exists(mp), f"{d} is not a finished route cache: run `lane_route_data.py build`"
        self.manifest = json.load(open(mp))
        assert self.manifest["version"] == VERSION, f"{d}: version {self.manifest['version']}, expected {VERSION}: rebuild"
        assert self.manifest["params"] == cache_params(), f"{d}: built with other constants {self.manifest['params']}, now {cache_params()}: rebuild"
        assert lane_gt is None or self.manifest["lane_gt"] == lane_gt, f"{d}: built for the lane GT {self.manifest['lane_gt']}, the token cache has {lane_gt}: rebuild"
        self.tokens = json.load(open(os.path.join(d, "tokens.json")))
        assert tokens_sha(self.tokens) == self.manifest["tokens_sha"] and len(self.tokens) == self.manifest["frames"], f"{d}: the frame list does not match its manifest"
        self.a = {n: np.load(os.path.join(d, n + ".npy"), mmap_mode="r") for n in ARRAYS}

    def __len__(self):
        return len(self.tokens)

    def labels(self, i):
        n = int(self.a["n_nodes"][i])
        return np.array(self.a["on_road"][i, :n]), np.array(self.a["on_lane"][i, :n])

    def labels_id(self, i):
        """the diagnostic road-id label of the table nodes of frame i"""
        return np.array(self.a["on_road_id"][i, :int(self.a["n_nodes"][i])])

    def polys(self, i):
        fe = self.a["frame_poly_end"]
        p0, p1 = (int(fe[i - 1]) if i else 0), int(fe[i])
        pe = self.a["poly_end"]
        out = []
        for p in range(p0, p1):
            v0, v1 = (int(pe[p - 1]) if p else 0), int(pe[p])
            out.append((np.array(self.a["poly_xy"][v0:v1]), np.array(self.a["poly_cnt"][v0:v1])))
        return out

    def exact(self, i):
        m = int(self.a["rt_n"][i])
        return check_keys(dict(pos=np.array(self.a["rt_pos"][i, :m]), dir=np.array(self.a["rt_dir"][i, :m]), count=np.array(self.a["rt_count"][i, :m]).astype(np.int64), valid=np.ones(m, bool)))


def route_dir(split, limit=None, cache=CACHE):
    return os.path.join(cache, split + (f"_first{limit}" if limit else ""))


def rows_guard(split, rows):
    """the TEST guard of what is read: every row is a row of the cache's frames.json AND of data/splits.json[split] (the poses and feature files also hold TEST rows): VAL = data/splits.json['val'] exactly and in order (873), TRAIN = the set of splits['train'] (8414)"""
    sp = json.load(open(os.path.join(ROOT, "data", "splits.json")))[split]
    assert set(int(r) for r in rows) <= set(int(r) for r in sp) and len(rows) == len(sp), f"the rows of the {split} cache are not data/splits.json['{split}']"
    LPD.check_rows(split, rows)


def build(split, seed=0, ident="e292_a", limit=None, cache=CACHE, log=print):
    """the route cache of a split from the frames of the E365 cache of B' (`ident`, `seed`; every cache of the split has the same frames) and the lane GT of this checkout -> the directory (a finished one is left as it is). TEST guard: VAL = data/splits.json['val']"""
    import lane_gt_data as LD
    import lane_tok_data as LT
    out = route_dir(split, limit, cache)
    sig = os.path.basename(LD.lane_store()[1])
    mp = os.path.join(out, "manifest.json")
    if os.path.exists(mp):
        old = json.load(open(mp))
        if old.get("version") == VERSION and old.get("lane_gt") == sig:
            log(f"route cache {out} is finished ({old['frames']} frames)")
            return out
        log(f"route cache {out}: version {old.get('version')} / lane GT {old.get('lane_gt')} is stale (now {VERSION} / {sig}): rebuilt")
    tdir = LP.cache_dir(split, seed, ident)
    fr = json.load(open(os.path.join(tdir, "frames.json")))
    rows_guard(split, fr["rows"])
    scene = LPD.scene_ids(LPD.load_ext(tdir)["prev"])
    td = LT.TokData(split, res=2.0, feats=False)
    assert td.tokens == fr["tokens"], "the GT tables are not in the order of the cache"
    lane_gt = LD.lane_store()[0]
    n = min(limit, len(td)) if limit else len(td)
    recs = []
    for i in range(n):
        recs.append(frame_record(lane_gt.load(td.tokens[i]), td.table(i)))
        if (i + 1) % 500 == 0 or i + 1 == n:
            log(f"   {split}: {i + 1}/{n} frames", flush=True)
    man = write_cache(out, pack(recs, scene[:n]), td.tokens[:n], dict(split=split, lane_gt=sig, source_cache=os.path.relpath(tdir, ROOT)))
    log(f"route cache {out}: {n} frames, {man['frames_with_route']} with a route, {man['decision_frames']} decision frames")
    return out


# ------------------------------------------------------------------------------------------------------------------------------ the dataset
class RouteData(torch.utils.data.Dataset):
    """the frames of `lane_ptok_data.PTokData` (the same item) + `on_road` / `on_lane` / `on_road_id` (K,) int8, `has_route`, `decision` and, unless mode is None, `route` (mode of the module docstring); `tok` / `rc` are injected by the selftest only"""

    def __init__(self, split="train", seed=0, ident="e292_a", K=None, mode="error", init_seed=0, cache=LC.CACHE, dir=None, rdir=None, tok=None, rc=None):
        assert mode is None or mode in MODES, mode
        if tok is None:
            tok = LP.PTokData(split, seed, ident, K, 0.0, cache, dir)
            if dir is None:
                rows_guard(split, tok.rows)                       # the TEST guard: a VAL cache is data/splits.json['val'] exactly
        self.tok, self.rc = tok, rc or TokenRoute(rdir or route_dir(split), token_lane_gt(tok))
        assert list(self.tok.tokens) == self.rc.tokens, f"the route cache {self.rc.dir} is not the frame list of the token cache {self.tok.dir}"
        self.mode, self.init_seed, self.tokens = mode, int(init_seed), list(self.tok.tokens)
        self.scene, self.has = np.asarray(self.rc.a["scene"]), np.asarray(self.rc.a["has_route"], bool)
        self.plan = other_scene_plan(self.scene, self.has) if mode in ("wrong", "train") else None
        self.draws = torch.zeros(len(self.tokens), dtype=torch.int32).share_memory_() if mode == "train" else None       # the draw index of every frame (shared with the workers)

    def __len__(self):
        return len(self.tok)

    def polys(self, i):
        return self.rc.polys(i)

    def raw_route(self, i, mode=None):
        """the raw route (pos, dir, count; before the cap) of frame i in `mode` (default: the dataset's); `train` makes one draw (and counts it)"""
        mode, tok = mode or self.mode, self.tokens[i]
        if mode == "none":
            return EMPTY
        if mode == "exact":
            return sample_route(self.polys(i))
        if mode == "train":
            return self._draw(i)
        if mode == "wrong":
            assert self.plan is not None, "mode wrong needs the scene plan: build the dataset with mode wrong / train"
            return error_route(self.polys(int(self.plan[i])), self.tokens[self.plan[i]])
        base = error_route(self.polys(i), tok)
        if mode == "error":
            return base
        ang = token_rng(tok, "|gross").uniform(0, 2 * np.pi)
        return shifted(base, GROSS_M * np.array([np.cos(ang), np.sin(ang)]))

    def _draw(self, i):
        """one draw of the mix of the module docstring for frame i"""
        assert self.draws is not None, "the draws need a dataset of mode train (the counters and the scene plan)"
        n = int(self.draws[i])
        self.draws[i] = n + 1
        rng = np.random.RandomState([self.init_seed % 2 ** 32, zlib.crc32(self.tokens[i].encode()), n])
        cls, polys = draw_class(rng.rand()), self.polys(i)
        if cls == "exact":
            return sample_route(polys)
        if cls == "absent":
            return EMPTY
        if cls in ("error", "unknown"):
            raw = error_like(polys, rng, *TRAIN_SHIFT)
            return with_count(raw, 0) if cls == "unknown" else raw
        if rng.rand() < 0.5:                                              # gross: a rigid shift of 4 - 8 m ...
            return error_like(polys, rng, *GROSS_SHIFT)
        j = other_frame(self.scene, self.has, i, int(rng.randint(len(self.tokens))))        # ... or the route of a frame of another scene
        return error_like(self.polys(j), rng, *TRAIN_SHIFT)

    def table(self, i, mode=None):
        return self.rc.exact(i) if (mode or self.mode) == "exact" else finish(self.raw_route(i, mode))

    def token_labels(self, i, gt_id):
        """(on_road, on_lane, on_road_id) (K,) int8 of the tokens of frame i: the label of the GT node a token is assigned to (`gt_id` of the E365 cache), -1 = no node"""
        got = np.asarray(gt_id) >= 0
        out = []
        for lab in self.rc.labels(i) + (self.rc.labels_id(i),):
            t = -np.ones(len(gt_id), np.int8)
            t[got] = lab[np.asarray(gt_id)[got]] if len(lab) else t[got]
            out.append(t)
        return out

    def __getitem__(self, i):
        out = self.tok[i]
        out["on_road"], out["on_lane"], out["on_road_id"] = self.token_labels(i, out["label"]["gt_id"])
        out["has_route"], out["decision"] = bool(self.has[i]), bool(self.rc.a["decision"][i])
        if self.mode is not None:
            out["route"] = self.table(i)
        return out


def collate(b):
    """`lane_ptok_data.collate` + the labels, the frame flags and the padded routes of the items that have them"""
    out = LP.collate(b)
    out.update(on_road=torch.from_numpy(np.stack([o["on_road"] for o in b])), on_lane=torch.from_numpy(np.stack([o["on_lane"] for o in b])), on_road_id=torch.from_numpy(np.stack([o["on_road_id"] for o in b])), has_route=torch.tensor([o["has_route"] for o in b]),
               decision=torch.tensor([o["decision"] for o in b]))
    if "route" in b[0]:
        out["route"] = collate_routes([o["route"] for o in b])
    return out


def loader(split="train", seed=0, bs=8, workers=4, init_seed=0, mode="auto", ident="e292_a", K=None, shuffle=None, cache=LC.CACHE, dir=None, rdir=None):
    """as `lane_ptok_data.loader` (the same sampler, batch size, workers) + the route and the labels; mode "auto": TRAIN `train`, VAL `error`; None: labels only, no route key; "absent" = "none" (the no-map arm)"""
    shuffle = split == "train" if shuffle is None else shuffle
    mode = ("train" if split == "train" else "error") if mode == "auto" else ("none" if mode == "absent" else mode)
    ds = RouteData(split, seed, ident, K, mode, init_seed, cache, dir, rdir)
    return ds, torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers, collate_fn=collate, drop_last=shuffle, persistent_workers=workers > 0, prefetch_factor=2 if workers > 0 else None)


# ------------------------------------------------------------------------------------------------------------------------------ loadercheck: the defaults of `loader` on the real caches
def loader_check(seed=0, ident="e292_a", cache=LC.CACHE, log=print):
    """the defaults of `loader` on the real caches: TRAIN = shuffled, drop_last, mode train (6 batches of 8 come from >= 4 scenes; some routes are the exact one, many are not); VAL = in order, not dropped, mode error (the route of a frame is its fixed error route)"""
    torch.manual_seed(0)
    ds, dl = loader("train", seed, bs=8, workers=0, init_seed=0, ident=ident, cache=cache)
    assert ds.mode == "train" and dl.drop_last and type(dl.sampler).__name__ == "RandomSampler", f"TRAIN default: mode {ds.mode}, drop_last {dl.drop_last}, sampler {type(dl.sampler).__name__}"
    at = {t: i for i, t in enumerate(ds.tokens)}
    scenes, same, differ, empty, n = set(), 0, 0, 0, 0

    def route_of_item(b, j):
        v = int(b["route"]["valid"][j].sum())
        return {k: b["route"][k][j][:v].numpy() for k in ("pos", "dir", "count")}, v
    for _, b in zip(range(6), dl):
        assert len(b["token"]) == 8 and b["route"]["pos"].shape[0] == 8
        for j, t in enumerate(b["token"]):
            i = at[t]
            scenes.add(int(ds.scene[i]))
            got, v = route_of_item(b, j)
            ex = ds.rc.exact(i)
            eq = v == len(ex["valid"]) and all(np.array_equal(got[k], ex[k]) for k in ("pos", "dir", "count"))
            same, differ, empty, n = same + int(eq), differ + int(not eq), empty + int(v == 0 and len(ex["valid"]) > 0), n + 1
    assert len(scenes) >= 4, f"6 shuffled batches of TRAIN come from {len(scenes)} scenes only"
    assert same >= 1 and differ >= n // 3, f"mode train: {same} of {n} routes are the exact one, {differ} differ: the draws of the train mix are missing"
    dv, dlv = loader("val", seed, bs=8, workers=0, ident=ident, cache=cache)
    assert dv.mode == "error" and dv.draws is None and not dlv.drop_last and type(dlv.sampler).__name__ == "SequentialSampler", f"VAL default: mode {dv.mode}, drop_last {dlv.drop_last}, sampler {type(dlv.sampler).__name__}"
    b = next(iter(dlv))
    assert b["token"] == dv.tokens[:8], "VAL batches are in the order of the cache"
    for j, t in enumerate(b["token"]):
        got, v = route_of_item(b, j)
        want = finish(error_route(dv.polys(j), t))
        assert v == len(want["valid"]) and all(np.array_equal(got[k], want[k]) for k in ("pos", "dir", "count")), f"VAL frame {j}: the route is not its fixed error route"
    log(f"loader defaults ok: TRAIN shuffled, drop_last, mode train: {n} frames of 6 batches from {len(scenes)} scenes, {same} with the exact route, {differ} others ({empty} without any sample); VAL in order, mode error, the fixed error routes")
    return dict(train_scenes=len(scenes), exact=same, other=differ, empty=empty)


# ------------------------------------------------------------------------------------------------------------------------------ check: the guards, the leak check, the alignment, the cache against a recompute
def train_guard(rows, tokens, splits, train_tokens, val_tokens):
    """the TRAIN guard: the rows of the TRAIN cache are exactly `data/splits.json['train']`, its frame tokens are those the train feature file holds at these rows, and none of them is the token of a VAL or TEST frame (`splits['val']` and `splits['test']` index
    the VAL feature file: another index space, so the disjointness is checked on the sample tokens) -> dict(train_frames, val_test_frames)"""
    assert len(set(rows)) == len(rows) and sorted(int(r) for r in rows) == sorted(int(r) for r in splits["train"]), "the rows of the TRAIN cache are not data/splits.json['train']"
    assert all(train_tokens[int(r)] == t for r, t in zip(rows, tokens)), "the tokens of the TRAIN cache are not those of the train feature file at its rows"
    held = {val_tokens[int(r)] for r in list(splits["val"]) + list(splits["test"])}
    both = held & set(tokens)
    assert not both, f"{len(both)} TRAIN frames are VAL / TEST frames (data/splits.json val + test)"
    return dict(train_frames=len(rows), val_test_frames=len(held))


def guards(seed=0, ident="e292_a", cache=LC.CACHE, log=print):
    """the TEST guards on the frame lists of the E365 caches: VAL = data/splits.json['val'] (873), TRAIN = data/splits.json['train'] (8414) and none of its frames a VAL / TEST frame -> dict"""
    import p16_data as PD
    splits = json.load(open(os.path.join(ROOT, "data", "splits.json")))
    tok_files = [json.load(open(f"{PD.D}/feats/{part}_tokens.json")) for part in ("train", "val")]
    res = {}
    for split in ("val", "train"):
        fr = json.load(open(os.path.join(LP.cache_dir(split, seed, ident, None, cache), "frames.json")))
        rows_guard(split, fr["rows"])
        assert len(fr["rows"]) == EXPECT[split], f"{split}: {len(fr['rows'])} frames, expected {EXPECT[split]}"
        if split == "train":
            res["train"] = train_guard(fr["rows"], fr["tokens"], splits, *tok_files)
            log(f"TRAIN guard: the {res['train']['train_frames']} TRAIN frames are data/splits.json['train'] and none is one of the {res['train']['val_test_frames']} VAL / TEST frames")
        else:
            res["val"] = dict(val_frames=len(fr["rows"]))
            log(f"VAL guard: the {len(fr['rows'])} VAL frames are data/splits.json['val'] in its order; the TEST rows are never read")
    return res


def check(n=None, seed=0, ident="e292_a", cache=LC.CACHE, log=print, splits=("val", "train")):
    """`guards`; for the VAL and the TRAIN cache (when built) n evenly spaced frames (default all): the node table's pieces lie on the frame's pieces (every node within 1 mm of a vertex of its piece), the cached route tokens equal those made from the cached polylines
    and from a recompute of the frame, every table of every read-out mode passes the leak check, the labels of a recompute equal the cached ones; the truncation counts -> dict"""
    import lane_gt_data as LD
    import lane_tok_data as LT
    res = dict(n=n, guards=guards(seed, ident, cache, log))
    lane_gt = LD.lane_store()[0]
    for split in splits:
        if not os.path.exists(os.path.join(route_dir(split), "manifest.json")):
            log(f"{split}: no route cache built, the cache part of the check is skipped")
            continue
        rc, td = TokenRoute(route_dir(split)), LT.TokData(split, res=2.0, feats=False)
        assert rc.tokens == td.tokens, f"{split}: the route cache is not in the order of the cache"
        ds = RouteData(split, seed, ident, None, "train", cache=cache, rc=rc)
        idx = np.arange(len(rc)) if n is None else np.linspace(0, len(rc) - 1, n).astype(int)
        worst = 0.0
        for i in idx:
            i = int(i)
            g, t = lane_gt.load(rc.tokens[i]), td.table(i)
            for k in np.unique(t["piece"]):                                                          # the piece index of a node is the index into g['pieces']
                q, P = t["pos"][t["piece"] == k], np.asarray(g["pieces"][k]["pts"], float)
                worst = max(worst, float(cKDTree(P).query(q)[0].max()))
            rec = frame_record(g, t)
            ron, rln = rc.labels(i)
            assert np.array_equal(ron, rec["on_road"]) and np.array_equal(rln, rec["on_lane"]) and np.array_equal(rc.labels_id(i), rec["on_road_id"]) and bool(rc.a["has_route"][i]) == rec["has_route"] and bool(rc.a["decision"][i]) == rec["decision"], f"{split} frame {i}: the cached labels differ from a recompute"
            assert len(rc.polys(i)) == len(rec["polys"]) and all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(rc.polys(i), rec["polys"])), f"{split} frame {i}: the cached polylines differ from a recompute"
            ex = finish(sample_route(rc.polys(i)))
            assert all(np.array_equal(ex[k], rc.exact(i)[k]) for k in KEYS), f"{split} frame {i}: the cached exact tokens differ from the polylines'"
            for mode in MODES:
                if mode != "exact":
                    check_keys(ds.table(i, mode))                                                    # the leak check of every table of every mode
        assert worst < 1e-3, f"{split}: a node is {worst:.4f} m from its piece: the table's piece index is not the index into g['pieces']"
        res[split] = dict(frames_checked=len(idx), node_to_piece_max_m=worst, **{k: rc.manifest[k] for k in ("frames", "frames_with_route", "frames_without_route", "decision_frames", "route_tokens_mean", "route_tokens_max", "frames_truncated")})
        log(f"{split}: {len(idx)} frames checked (labels, polylines, exact tokens = a recompute; node-to-piece max {worst:.2e} m; every table of every mode has the keys {KEYS}); route tokens mean {res[split]['route_tokens_mean']:.1f} max {res[split]['route_tokens_max']}, "
            f"truncated at {MAX_SAMPLES}: {res[split]['frames_truncated']} frames")
    return res


# ------------------------------------------------------------------------------------------------------------------------------ fixture: the card's section 1 (a) - (d) on real frames
FIX_TOL = dict(route_nodes_in_band=0.99, sd_p90=2.5, piece_p90=1.0, lateral_p90=2.0, lateral_min=0.5, share_parallel_off=0.95, equal_tol=0.2, share_single_equal=0.99)
HOWS = ("true", "mirror", "other_frame", "reversed")                                        # the driven path and the wrong ones the fixture must reject


def path_of(how, path, other):
    """the driven path as `how` presents it: the true one, mirrored (x -> -x), the driven path of ANOTHER frame, the reversed path"""
    return dict(true=path, mirror=path * np.array([-1.0, 1.0]), other_frame=other, reversed=path[::-1]).get(how)


def fixture_checks(s, tol=FIX_TOL):
    """the rule of the fixture on its numbers (dict of `fixture`) -> name -> bool. (a) the driven path of the frames lies, pooled over the frames, within 2.5 m of the SD route polyline and within 1.0 m of a route piece at p90 (the heading of the path must agree with the route's;
    a frame without a route counts as infinitely far), every wrong path FAILS that; (b) every on_lane node is an on_road node, the on_lane share does not exceed the on_road share, every node of a route piece is on_road; (c) a route table has exactly the 4 keys, the SD route is
    within 2 m of the lane centre path, differs from it on the frames with >= 2 parallel lanes and equals it (within 0.2 m: the curve resolution 0.125 m + the chord sag) where there is one lane; (d) the guards ran (the fixture reads only rows of the cache: VAL = data/splits.json['val'])"""
    ok = {"(a) the driven path is near the SD route (pooled p90 <= 2.5 m) and near a route piece (p90 <= 1.0 m)": s["dist"]["true"]["sd_p90"] <= tol["sd_p90"] and s["dist"]["true"]["piece_p90"] <= tol["piece_p90"]}
    for how in HOWS:
        if how != "true":
            ok[f"(a) the {how} path fails that"] = not (s["dist"][how]["sd_p90"] <= tol["sd_p90"] and s["dist"][how]["piece_p90"] <= tol["piece_p90"])
    ok["(b) every on_lane node is an on_road node"] = s["lane_not_road"] == 0
    ok["(b) the on_lane share does not exceed the on_road share"] = s["share_on_lane"] <= s["share_on_road"]
    ok["(b) every node of a route piece is on_road"] = s["route_piece_nodes_not_road"] == 0
    ok["(b) the band alone holds the nodes of the route pieces (>= 99 %)"] = s["share_route_nodes_in_band"] >= tol["route_nodes_in_band"]
    ok["(c) the route table has exactly the 4 keys"] = s["keys_ok"]
    ok["(c) the SD route is within 2 m of the lane centre path (p90)"] = s["lateral_p90"] <= tol["lateral_p90"]
    ok["(c) the SD route differs from the lane path where >= 2 lanes run in parallel"] = s["share_parallel_off"] >= tol["share_parallel_off"]
    ok["(c) the SD route equals the lane path where there is one lane"] = s["share_single_equal"] >= tol["share_single_equal"]
    ok["(d) the guards ran"] = s["guards"]
    return ok


def fixture(split="val", n=200, seed=0, ident="e292_a", cache=LC.CACHE, log=print):
    """the module docstring's fixture on n evenly spaced frames of the split: see `fixture_checks`; passed = every check -> dict. Needs the route cache of the split (labels) and the E365 cache of B' (the covered GT nodes). Reported next to it: the per-frame distribution
    of the path-to-route distance and the agreement of the tier-1 route with `viterbi_route` (the pieces the path follows)"""
    import lane_gt_data as LD
    lane_gt = LD.lane_store()[0]
    rc = TokenRoute(route_dir(split))
    c = LC.TokCache(LP.cache_dir(split, seed, ident, None, cache))
    assert rc.tokens == c.tokens, "the route cache is not in the order of the token cache"
    rows_guard(split, c.rows)
    N = len(rc)
    idx = np.linspace(0, N - 1, min(n, N)).astype(int)
    d = {h: dict(sd=[], piece=[], frame=[]) for h in HOWS}
    lat, par_off, single_eq, n_par, n_single, n_route, jacc, keys_ok = [], 0, 0, 0, 0, 0, [], True
    road_n = lane_n = id_n = cov_n = lane_not_road = piece_nodes_not_road = lane_nodes = lane_nodes_in_band = 0
    for i in idx:
        i = int(i)
        g, gother = lane_gt.load(c.tokens[i]), lane_gt.load(c.tokens[(i + OTHER) % N])
        route = route_of(g)
        polys = route_polys(g, route) if route else []
        sd, piece_curve = to_curve([p for p, _ in polys]), to_curve([np.asarray(g["pieces"][k]["pts"], float) for k in route])
        for how in HOWS:
            dense = densify(path_of(how, np.asarray(g["ego_path"]["pts"], float), np.asarray(gother["ego_path"]["pts"], float)))
            dirs, ins = unit_tangents(dense), (np.abs(dense) <= g["half"]).all(1)
            dense, dirs = dense[ins], dirs[ins]
            ds_, dp_ = curve_dist(sd, dense, dirs), curve_dist(piece_curve, dense, dirs)
            d[how]["sd"].append(ds_)
            d[how]["piece"].append(dp_)
            d[how]["frame"].append(float(np.percentile(ds_, 90, method="higher")) if len(ds_) else float("nan"))
        if route:
            n_route += 1
            vit = set(viterbi_route(g))
            jacc.append(len(vit & set(route)) / max(len(vit | set(route)), 1))
            keys_ok = keys_ok and sorted(rc.exact(i)) == sorted(KEYS) and sorted(finish(error_route(polys, c.tokens[i]))) == sorted(KEYS)
            lane_curve = to_curve([np.asarray(g["pieces"][k]["pts"], float) for k in route])
            dev = np.concatenate([curve_dist(lane_curve, sample_route([pc])[0]) for pc in polys]) if polys else np.zeros(0)
            lat.append(dev)
            if len(dev):
                if max(float(np.max(cn)) for _, cn in polys) >= 2:
                    n_par += 1
                    par_off += int(dev.max() > FIX_TOL["lateral_min"])
                else:
                    n_single += 1
                    single_eq += int(dev.max() <= FIX_TOL["equal_tol"])
            ron, rln = rc.labels(i)
            gt = np.array(c.f["gt_id"][i])
            cov = np.unique(gt[np.array(c.f["valid"][i]) & (np.array(c.f["e_p"][i], float) >= 0.5) & (gt >= 0)])
            road_n, lane_n, id_n, cov_n = road_n + int(ron[cov].sum()), lane_n + int(rln[cov].sum()), id_n + int(rc.labels_id(i)[cov].sum()), cov_n + len(cov)
            lane_not_road += int(((rln == 1) & (ron != 1)).sum())
            tab = LT_table(split, i)
            route_nodes = np.isin(tab["piece"], route)
            piece_nodes_not_road += int((route_nodes & (ron != 1)).sum())
            lane_nodes, lane_nodes_in_band = lane_nodes + int(route_nodes.sum()), lane_nodes_in_band + int((route_nodes & road_band(polys, tab["pos"], tab["dirv"])).sum())
    cat = lambda xs: np.concatenate(xs)  # noqa: E731
    fr = lambda h: np.array([x for x in d[h]["frame"] if x == x])  # noqa: E731
    res = dict(split=split, n=len(idx), seed=seed, tolerances=FIX_TOL, dist={h: dict(sd_p90=float(np.percentile(cat(d[h]["sd"]), 90, method="higher")), piece_p90=float(np.percentile(cat(d[h]["piece"]), 90, method="higher")), sd_median=float(np.percentile(cat(d[h]["sd"]), 50, method="higher")),
                                                                                     points=int(sum(len(x) for x in d[h]["sd"])), far_points=int((~(cat(d[h]["sd"]) <= FIX_TOL["sd_p90"])).sum()),
                                                                                     frame_p90_percentiles={q: float(np.percentile(fr(h), q, method="higher")) for q in (10, 50, 90, 100)}, frames_within=float((fr(h) <= FIX_TOL["sd_p90"]).mean())) for h in HOWS},
               frames_with_route=n_route, share_on_road=road_n / max(cov_n, 1), share_on_lane=lane_n / max(cov_n, 1), share_on_road_id=id_n / max(cov_n, 1), share_route_nodes_in_band=lane_nodes_in_band / max(lane_nodes, 1), covered_nodes=cov_n, lane_not_road=lane_not_road, route_piece_nodes_not_road=piece_nodes_not_road, keys_ok=bool(keys_ok),
               lateral_p90=float(np.percentile(cat(lat), 90)), share_parallel_off=par_off / max(n_par, 1), frames_parallel=n_par, share_single_equal=single_eq / max(n_single, 1), frames_single=n_single,
               viterbi_agreement_mean=float(np.mean(jacc)), viterbi_agreement_p10=float(np.percentile(jacc, 10)), viterbi_identical=float(np.mean(np.array(jacc) == 1.0)), guards=True)
    ok = res["checks"] = fixture_checks(res)
    res["passed"] = all(ok.values())
    log(f"fixture {split} {N} frames, {len(idx)} checked ({n_route} with a route)")
    for h in HOWS:
        v = res["dist"][h]
        q = v["frame_p90_percentiles"]
        log(f"  path {h:<12} to the SD route (pooled over {v['points']} points; heading must agree): p90 {v['sd_p90']:.2f} m, median {v['sd_median']:.2f}; to a route piece p90 {v['piece_p90']:.2f} m; points beyond 2.5 m {v['far_points']}; "
            f"per-frame p90 of the distance: p10 {q[10]:.2f} median {q[50]:.2f} p90 {q[90]:.2f} max {q[100]:.2f} m, {v['frames_within']:.3f} of the frames within 2.5 m")
    log(f"  covered GT nodes {cov_n}: on_road (band) {res['share_on_road']:.3f}, on_road_id {res['share_on_road_id']:.3f}, on_lane {res['share_on_lane']:.3f}; route-piece nodes inside the band by themselves {res['share_route_nodes_in_band']:.4f}; on_lane but not on_road {lane_not_road}; route-piece nodes not on_road {piece_nodes_not_road}")
    log(f"  SD route to the lane centre path: p90 {res['lateral_p90']:.2f} m; frames with >= 2 parallel lanes {n_par}: differs by > 0.5 m on {res['share_parallel_off']:.3f}; one lane only {n_single}: equal within 0.2 m on {res['share_single_equal']:.3f}")
    log(f"  cross-check: the tier-1 route against `viterbi_route` (Jaccard of the piece sets): mean {res['viterbi_agreement_mean']:.3f}, p10 {res['viterbi_agreement_p10']:.3f}, identical on {res['viterbi_identical']:.3f} of the frames")
    for k, v in ok.items():
        log(f"  {'ok  ' if v else 'FAIL'} {k}")
    log("PASS" if res["passed"] else "FAIL")
    return res


_TABLES = {}


def LT_table(split, i):
    """the E357 table of frame i (the tables of the cache, 2 m lattice)"""
    import lane_tok_data as LT
    if split not in _TABLES:
        _TABLES[split] = LT.TokData(split, res=2.0, feats=False)
    return _TABLES[split].table(i)


# ------------------------------------------------------------------------------------------------------------------------------ stats
def stats(split="val", seed=0, ident="e292_a", cache=LC.CACHE, log=print):
    """the numbers of the route cache of a split against the E365 cache of B' (`ident`, `seed`): the share of frames with a route, the label shares of the GT nodes covered by a token (valid, assigned by gt_id) and by a DELIVERED token (existence >= 0.5), the route tokens, the
    decision frames, the covered-node counts on decision frames -> dict"""
    rc = TokenRoute(route_dir(split))
    c = LC.TokCache(LP.cache_dir(split, seed, ident, None, cache))
    assert rc.tokens == c.tokens, "the route cache is not in the order of the token cache"
    a = rc.a
    has, dec = np.asarray(a["has_route"], bool), np.asarray(a["decision"], bool)
    acc = {k: np.zeros(4, np.int64) for k in ("valid", "deliv", "deliv_dec")}                         # nodes, on_road, on_lane, on_road_id
    for i in np.flatnonzero(has):
        gt, val, ep = np.array(c.f["gt_id"][i]), np.array(c.f["valid"][i]), np.array(c.f["e_p"][i], float)
        ron, rln, rid = rc.labels(i) + (rc.labels_id(i),)
        for key, m in (("valid", val & (gt >= 0)), ("deliv", val & (ep >= 0.5) & (gt >= 0)), ("deliv_dec", val & (ep >= 0.5) & (gt >= 0) & bool(dec[i]))):
            cov = np.unique(gt[m])
            acc[key] += [len(cov), int(ron[cov].sum()), int(rln[cov].sum()), int(rid[cov].sum())]
    share = lambda v: dict(nodes=int(v[0]), on_road=float(v[1] / max(v[0], 1)), on_lane=float(v[2] / max(v[0], 1)), on_road_id=float(v[3] / max(v[0], 1)))  # noqa: E731
    n = len(rc)
    res = dict(split=split, frames=n, frames_with_route=int(has.sum()), share_with_route=float(has.mean()), frames_without_route=int((~has).sum()), route_pieces_mean=float(a["n_route_pieces"][has].mean()), decision_frames=int(dec.sum()),
               scenes=int(len(set(np.asarray(a['scene']).tolist()))), scenes_with_fork=int(len(set(np.asarray(a['scene'])[dec].tolist()))), share_decision=float(dec.mean()), covered=share(acc["valid"]), delivered=share(acc["deliv"]), delivered_on_decision_frames=share(acc["deliv_dec"]), route_tokens_mean=float(np.asarray(a["rt_n"])[has].mean()), route_tokens_max=int(np.asarray(a["rt_n"]).max()),
               frames_truncated=int((np.asarray(a["rt_cut"]) > 0).sum()), frames_with_route_but_no_token=int((has & (np.asarray(a["rt_n"]) == 0)).sum()), cache=os.path.relpath(c.dir, ROOT))
    log(f"{split}: {n} frames, route on {res['frames_with_route']} ({res['share_with_route']:.4f}), without {res['frames_without_route']}; route pieces per frame {res['route_pieces_mean']:.2f}; fork (decision) frames {res['decision_frames']} ({res['share_decision']:.3f}) in {res['scenes_with_fork']} of {res['scenes']} scenes")
    for k in ("covered", "delivered", "delivered_on_decision_frames"):
        log(f"  {k:<30} GT nodes {res[k]['nodes']:7d}: on_road (band) {res[k]['on_road']:.4f}, on_road_id {res[k]['on_road_id']:.4f}, on_lane {res[k]['on_lane']:.4f}")
    log(f"  route tokens per frame (with a route) mean {res['route_tokens_mean']:.1f}, max {res['route_tokens_max']}, truncated at {MAX_SAMPLES}: {res['frames_truncated']} frames, a route but no token: {res['frames_with_route_but_no_token']}")
    return res


# ------------------------------------------------------------------------------------------------------------------------------ selftest
def refuses(fn, text):
    """fn must fail with an AssertionError whose message holds `text`; a call that is accepted, or that fails in another way, is an assertion failure of the test"""
    try:
        fn()
    except AssertionError as e:
        assert text in str(e), (text, str(e))
        return
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"{type(e).__name__} instead of a refusal ({text}): {e}")
    raise AssertionError(f"accepted although it must be refused ({text})")


def line(a, b, step=0.5):
    """a straight polyline from a to b with the vertices `step` m apart"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a + (b - a) * np.linspace(0.0, 1.0, max(int(round(np.linalg.norm(b - a) / step)), 1) + 1)[:, None]


def arc(c, r, a0, a1, step=0.5):
    """a circular arc about the centre c, radius r, from the angle a0 to a1 (rad), the vertices about `step` m apart"""
    th = np.linspace(a0, a1, max(int(round(abs(a1 - a0) * r / step)), 1) + 1)
    return np.asarray(c, float) + r * np.stack([np.cos(th), np.sin(th)], 1)


def toy_graph(fork_at=10.0, tiered=True):
    """a toy lane GT frame (the keys `route_of`, `node_labels`, `decision_frame` and `viterbi_route` read; the lane frame, x right, y forward): the lane A runs north from y = -40 to `fork_at` and forks into B (straight on to y = 50), C (a right turn of radius 8 m, then
    east to x = 50) and C2 (a connector 1.0 m beside C); D is the lane beside A (3.5 m to the right, the same road), G the next lane (7 m, its own road id but listed in the road of A, as a lane that leaves the carriageway), E the opposite lane 2 m left of A
    (heading south), F a crossing road far away. The driven path: A from y = -25, then C, ending before x = 40. Tiers: 1 on A and C (the ego lane), 3 elsewhere (`tiered` False: no tier-1 point at all)"""
    f = fork_at
    C = np.concatenate([arc([8.0, f], 8.0, np.pi, np.pi / 2), line([8.0, f + 8.0], [50.0, f + 8.0])[1:]])            # from (0, f) heading north, turning right
    pts = dict(A=line([0.0, -40.0], [0.0, f]), B=line([0.0, f], [0.0, 50.0]), C=C, C2=C + [0.0, 1.0], D=line([3.5, -40.0], [3.5, f]), E=line([-2.0, 50.0], [-2.0, -40.0]), F=line([-50.0, 30.0], [-20.0, 30.0]), G=line([7.0, -40.0], [7.0, f]))
    succ = dict(A=["B", "C", "C2"], B=[], C=[], C2=[], D=[], E=[], F=[], G=[])
    pred = dict(A=[], B=["A"], C=["A"], C2=["A"], D=[], E=[], F=[], G=[])
    road = dict(A="rA", B="rB", C="rC", C2="rC2", D="rA", E="rE", F="rF", G="rG")
    pieces = [dict(id=k, pts=pts[k], succ=succ[k], pred=pred[k], road=road[k], tier=np.full(len(pts[k]), 1 if (tiered and k in "AC") else 3)) for k in ("A", "B", "C", "C2", "D", "E", "F", "G")]
    roads = [dict(id="rA", pieces=["A", "D", "G"]), dict(id="rB", pieces=["B"]), dict(id="rC", pieces=["C"]), dict(id="rC2", pieces=["C2"]), dict(id="rE", pieces=["E"]), dict(id="rF", pieces=["F"]), dict(id="rG", pieces=["G"])]
    p = np.concatenate([line([0.0, -25.0], [0.0, f])[:-1], C])
    return dict(pieces=pieces, roads=roads, half=50.0, ego_path=dict(pts=p[np.linalg.norm(p - [50.0, f + 8.0], axis=1) > 10.0].astype(np.float32), back_m=25.0, ahead_m=55.0))


def toy_table(g, frac=(0.2, 0.5, 0.8)):
    """the GT node table of a toy frame (the keys `node_labels` reads): three nodes per piece at 20 / 50 / 80 % of its vertices, with the heading of the piece there (piece-major: node 3 k + j is the j-th node of piece k)"""
    pos, dirv, piece = [], [], []
    for k, p in enumerate(g["pieces"]):
        pts = np.asarray(p["pts"], float)
        tg = unit_tangents(pts)
        for f in frac:
            j = int(f * (len(pts) - 1))
            pos.append(pts[j])
            dirv.append(tg[j])
            piece.append(k)
    return dict(pos=np.array(pos), dirv=np.array(dirv), piece=np.array(piece))


def toy_world(n=12):
    """a toy cache world: n frames of the toy graph in 3 scenes of 4 frames; frame k has its fork at 10 + k m, the frames 5 and 9 have no tier-1 point (no route); -> (graphs, tokens, scene, records, arrays)"""
    graphs, recs = [], []
    for k in range(n):
        g = toy_graph(10.0 + k, tiered=k not in (5, 9))
        graphs.append(g)
        recs.append(frame_record(g, toy_table(g)))
    scene = np.repeat(np.arange(3), n // 3)
    return graphs, [f"toy{k}" for k in range(n)], scene, recs, pack(recs, scene)


class ToyTok:
    """the stand-in of `PTokData` for the selftest: tokens, one item per frame with the GT node of every token (token t of frame i is assigned to node 5 t mod 24 of the 24-node table: all residues mod 3, so all positions in a piece; -1 for t >= 16)"""

    def __init__(self, tokens, K=20):
        self.tokens, self.rows, self.K, self.dir = list(tokens), list(range(len(tokens))), K, None

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        gt = np.array([(5 * t) % 24 if t < 16 else -1 for t in range(self.K)], np.int32)
        return dict(token=self.tokens[i], row=i, label=dict(gt_id=gt))


def toy_data(mode, rc, n=12, **kw):
    return RouteData(tok=ToyTok([f"toy{k}" for k in range(n)]), rc=rc, mode=mode, **kw)


def selftest():
    # the numbers of the card, pinned: a changed constant is a new card
    assert (STEP, MAX_SAMPLES, COUNT_MAX, GROSS_M, OTHER, MAX_D, AHEAD) == (2.0, 160, 5, 5.0, 97, 2.5, 40.0) and (ERROR_SHIFT, TRAIN_SHIFT, GROSS_SHIFT) == ((1.0, 2.0), (0.0, 2.0), (4.0, 8.0)) and abs(COS_MIN - np.cos(np.pi / 4)) < 1e-12
    assert MIX == (("exact", 0.25), ("error", 0.47), ("unknown", 0.08), ("absent", 0.10), ("gross", 0.10)) and (COST_SUCC, COST_JUMP) == (0.3, 3.0)
    # the route = the tier-1 pieces in driving order (A then C), whatever the path does: a stopped ego still has a route, a path that hugs C2 does not change it, a piece with ONE tier-1 point is a route piece, no tier-1 point = no route
    g = toy_graph()
    names = [p["id"] for p in g["pieces"]]
    assert [names[k] for k in route_of(g)] == ["A", "C"], [names[k] for k in route_of(g)]
    stopped = dict(g, ego_path=dict(pts=np.array([[0.0, 0.0], [0.0, 0.1]], np.float32)))
    assert [names[k] for k in route_of(stopped)] == ["A", "C"], "the route is the GT's tier-1, not what the path matches"
    hug = toy_graph()
    hug["ego_path"]["pts"] = (hug["ego_path"]["pts"] + [0.0, 1.0]).astype(np.float32)[10:]
    assert [names[k] for k in route_of(hug)] == ["A", "C"] and [names[k] for k in viterbi_route(hug)] == ["A", "C2"], "the cross-check follows the path (C2), the route does not"
    one = toy_graph()
    one["pieces"][5]["tier"][3] = 1
    assert [names[k] for k in route_of(one)] == ["A", "C", "E"] or [names[k] for k in route_of(one)] == ["A", "E", "C"] or sorted(names[k] for k in route_of(one)) == ["A", "C", "E"], "a piece with one tier-1 point is a route piece"
    assert route_of(toy_graph(tiered=False)) == [] and route_of(dict(pieces=[], roads=[], half=50.0, ego_path=dict(pts=np.zeros((0, 2))))) == []
    rev = toy_graph()
    rev["pieces"] = rev["pieces"][::-1]
    assert [rev["pieces"][k]["id"] for k in route_of(rev)] == ["A", "C"], "driving order = successor order, not the order of the list"
    gap = toy_graph()
    gap["pieces"][2]["pred"], gap["pieces"][0]["succ"] = [], ["B"]                           # no successor link A -> C: the order falls back to the position along the driven path
    assert [names[k] for k in route_of(gap)] == ["A", "C"], "pieces without a link are ordered by their place along the path"
    chain = dict(pieces=[dict(id=n_, pts=line([0.0, 10.0 * k], [0.0, 10.0 * k + 10.0]), succ=s_, pred=[], road=n_, tier=np.ones(21, int)) for k, (n_, s_) in enumerate([("P3", []), ("P1", ["P2"]), ("P2", ["P3"])])], roads=[], half=50.0,
                 ego_path=dict(pts=np.array([[0.0, 0.0]], np.float32)))                    # one path point: every piece has the same place, the successor links alone order them
    assert [chain["pieces"][k]["id"] for k in route_of(chain)] == ["P1", "P2", "P3"], "driving order = the successor chain"
    # the TEST guard of what is read: the rows of a cache are the rows of data/splits.json[split] (VAL exactly and in order, TRAIN as a set); a missing row, a TEST row, a foreign row is refused
    sp = json.load(open(os.path.join(ROOT, "data", "splits.json")))
    rows_guard("val", sp["val"])
    rows_guard("train", sp["train"])
    for bad in (sp["val"][:-1], sp["val"][::-1], sp["val"][:-1] + [sp["test"][0]], [sp["test"][0]] + sp["val"][1:]):
        refuses(lambda: rows_guard("val", bad), "splits.json")
    for bad in (sp["train"][:-1], sp["train"][:-1] + [max(sp["train"]) + 1]):
        refuses(lambda: rows_guard("train", bad), "splits.json")
    # the cross-check (`viterbi_route`): the heading cosine bound is 45 degrees (a lane 40 degrees off the path is matched, 50 degrees is not), the distance bound 2.5 m (2.4 m matches, 2.6 m does not)
    def one_lane(angle_deg, off):
        a = np.deg2rad(angle_deg)
        return dict(pieces=[dict(id="L", pts=line([off, -30.0], [off + 60.0 * np.sin(a), -30.0 + 60.0 * np.cos(a)]), succ=[], pred=[], road="r")], half=50.0, ego_path=dict(pts=line([0.0, -30.0], [0.0, 30.0])[::2]))
    assert viterbi_route(one_lane(40.0, 0.0)) == [0] and viterbi_route(one_lane(50.0, 0.0)) == [] and viterbi_route(one_lane(0.0, 2.4)) == [0] and viterbi_route(one_lane(0.0, 2.6)) == [], "match: heading within 45 degrees, distance within 2.5 m"
    assert viterbi_route(one_lane(0.0, 0.0), path=one_lane(0.0, 0.0)["ego_path"]["pts"][::-1]) == [], "a reversed path runs against the lane"
    assert viterbi_route(one_lane(0.0, 0.0), path=one_lane(0.0, 0.0)["ego_path"]["pts"][:1]) == [], "a path of one point has no route"
    assert [names[k] for k in viterbi_route(g)] == ["A", "C"], "the opposite lane (heading), D (distance), B and the connector beside C (the successor cost) are not taken"
    far = toy_graph()
    far["half"] = 5.0
    assert [names[k] for k in viterbi_route(far)] == ["A"], "points outside the window are not matched"
    assert successors(dict(pieces=[dict(id="a", succ=["b", "zz"]), dict(id="b", succ=[])])) == {(0, 1)}, "a successor outside the frame is ignored"
    # the band by hand: the exact route polyline x = 0 going north; 2 lanes = n * 1.75 + 0.5 = 4.0 m (inclusive, either side), 1 lane 2.25 m, 3 lanes 5.75 m, the count is clipped to 5 (9.25 m) and a count under 0.5 is 1 lane; the count is the one of the NEAREST route point
    yy = np.arange(-20.0, 20.01, 1.0)
    straight = np.stack([np.zeros(len(yy)), yy], 1)
    north, south = np.array([0.0, 1.0]), np.array([0.0, -1.0])
    band = lambda cnt, pts, d=north: road_band([(straight, np.broadcast_to(cnt, len(yy)))], pts, np.tile(d, (len(pts), 1))).tolist()  # noqa: E731
    assert band(2.0, [[4.0, 0.0], [4.05, 0.0], [-4.0, 0.0], [-4.05, 0.0], [0.0, 30.0]]) == [True, False, True, False, False], "2 lanes: 4.0 m inclusive on either side, nothing beyond the end of the route"
    assert band(1.0, [[2.25, 0.0], [2.3, 0.0]]) == [True, False] and band(3.0, [[5.75, 0.0], [5.8, 0.0]]) == [True, False] and band(7.0, [[9.25, 0.0], [9.3, 0.0]]) == [True, False] and band(0.2, [[2.25, 0.0], [2.3, 0.0]]) == [True, False], "1, 3, 7 (clipped to 5) lanes, a count under 0.5 is one lane"
    mixed = [(straight, np.where(yy < 0, 1.0, 3.0))]
    assert road_band(mixed, [[3.0, -10.0], [3.0, 10.0]], np.tile(north, (2, 1))).tolist() == [False, True], "the lane count of the nearest route point decides"
    assert band(2.0, [[0.0, 0.0]], np.array([np.sqrt(1 - 0.7001 ** 2), 0.7001])) == [True] and band(2.0, [[0.0, 0.0]], np.array([np.sqrt(1 - 0.6999 ** 2), 0.6999])) == [False], "the heading cosine bound is 0.7: 45.6 degrees is in, 45.8 is out"
    assert band(2.0, [[1.0, 0.0]], south) == [False], "a node against the direction of travel is not on the carriageway"
    assert road_band([], [[0.0, 0.0]], [north]).tolist() == [False] and road_band([(np.array([[0.0, 0.0]]), np.ones(1))], [[0.0, 0.0]], [north]).tolist() == [False] and road_band(mixed, np.zeros((0, 2)), np.zeros((0, 2))).shape == (0,), "no polyline / a polyline of one point / no node"
    # the labels on the toy frame: on_lane = a route piece (A, C); on_road = the band around the route or on_lane: D (beside A, 1.75 m from the centred route, 2 lanes: 4.0 m) and the connector C2 (1.0 m from C) are in the band, the opposite lane E (3.75 m, but against the direction),
    # G (5.25 m) and the branch B are not; on_road_id (diagnostic) = a piece in the piece list of a road that lists a route piece: A, D and G (the road of A lists them) and C, not C2 or B
    tab = toy_table(g)
    polys_g = route_polys(g, route_of(g))
    rep3 = lambda v: [x for x in v for _ in range(3)]  # noqa: E731
    on_road, on_lane, on_id = node_labels(g, route_of(g), tab, polys_g)
    assert on_lane.tolist() == rep3([1, 0, 1, 0, 0, 0, 0, 0]) and on_road.tolist() == rep3([1, 0, 1, 1, 1, 0, 0, 0]) and on_id.tolist() == rep3([1, 0, 1, 0, 1, 0, 0, 1]), (on_lane, on_road, on_id)
    assert node_labels(dict(g, roads=[]), route_of(g), tab, polys_g)[2].tolist() == on_lane.tolist(), "a route piece is on_road_id even when no road lists it"
    assert node_labels(g, route_of(g), tab, [])[0].tolist() == on_lane.tolist(), "a route piece is on_road even when the band holds nothing"
    refuses(lambda: node_labels(g, [0], dict(tab, piece=np.array([8])), polys_g), "no piece of the frame")
    assert node_labels(g, [0], dict(pos=np.zeros((0, 2)), dirv=np.zeros((0, 2)), piece=np.zeros(0, int)), polys_g)[0].shape == (0,)
    # the record: no route = every label -1; the route polylines are the centred pieces (A is beside D: count 2 on A, 1 on C)
    rec = frame_record(g, tab)
    assert rec["has_route"] and rec["decision"] and len(rec["polys"]) == 2 and (rec["polys"][0][1] == 2).all() and (rec["polys"][1][1] == 1).all() and abs(rec["polys"][0][0][:, 0].mean() - 1.75) < 1e-9, "A is centred between A and D"
    assert rec["on_road"].tolist() == on_road.tolist() and rec["on_lane"].tolist() == on_lane.tolist() and rec["on_road_id"].tolist() == on_id.tolist()
    none = frame_record(toy_graph(tiered=False), tab)
    assert not none["has_route"] and (none["on_road"] == -1).all() and (none["on_lane"] == -1).all() and (none["on_road_id"] == -1).all() and none["polys"] == [] and none["route"] == [] and not none["decision"] and len(none["on_road"]) == 24
    # decision frames = FORKS ONLY: the end of a route piece with more than one successor within 40 m in front of the ego along the route (40.0 inclusive); a fork behind the ego, a route piece without a fork, a fork piece that is not on the route, a merge: no
    assert decision_frame(g, route_of(g)) and decision_frame(toy_graph(40.0), route_of(toy_graph(40.0))) and not decision_frame(toy_graph(40.5), route_of(toy_graph(40.5))) and not decision_frame(g, [])
    assert not decision_frame(toy_graph(-5.0), route_of(toy_graph(-5.0))), "a fork behind the ego is no decision"
    assert not decision_frame(g, [2]) and not decision_frame(g, [1]) and decision_frame(g, [0]) and not decision_frame(g, [0], ahead=5.0), "a route piece without a fork / the fork piece off the route / a fork farther than `ahead`"
    gm = toy_graph()
    gm["pieces"][0]["pred"], gm["pieces"][0]["succ"] = ["X", "Y"], ["C"]                     # A has two predecessors and one successor and ends 10 m ahead: a merge, no fork
    assert not decision_frame(gm, route_of(gm)), "a merge creates no route ambiguity"
    gl = toy_graph(10.0)
    gl["pieces"][0]["pts"] = np.concatenate([line([0.0, -80.0], [0.0, -40.0], 0.5)[:-1], gl["pieces"][0]["pts"]])        # a longer lane A: the fork is still 10 m ahead (the length counted is from the ego on)
    assert decision_frame(gl, route_of(gl))
    # the pieces of a fork share their first metres: a branch that an earlier polyline covers (within 1 m, heading cosine >= 0.9) is cut out, the diverging rest is kept, an opposite-direction polyline is not covered, a run of one vertex is dropped
    main_ = (line([0.0, 0.0], [0.0, 20.0], 2.0), np.full(11, 2.0))
    branch = (np.concatenate([line([0.5, 0.0], [0.5, 10.0], 2.0), line([0.5, 10.0], [10.0, 20.0], 2.0)[1:]]), np.full(12, 1.0))
    kept = dedupe_polys([main_, branch])
    assert len(kept) == 2 and np.array_equal(kept[0][0], main_[0]) and len(kept[1][0]) == 7 and np.allclose(kept[1][0][0], [0.5 + 9.5 / 7, 10.0 + 10.0 / 7]) and (kept[1][1] == 1.0).all(), "the shared stretch is cut out, the diverging rest kept with its counts"
    assert len(dedupe_polys([main_, (main_[0][::-1], main_[1])])) == 2 and len(dedupe_polys([main_, main_])) == 1 and dedupe_polys([main_, main_])[0][0].shape == (11, 2) and dedupe_polys([]) == [], "the opposite direction is another road; an identical polyline is dropped"
    assert len(dedupe_polys([main_, (np.array([[0.5, 4.0], [0.5, 5.0], [8.0, 5.0], [9.0, 5.0]]), np.ones(4))])) == 2 and len(dedupe_polys([main_, (np.array([[0.5, 4.0], [0.5, 5.0]]), np.ones(2))])) == 1, "a covered pair is dropped, a stretch of 2 vertices that is not covered is kept"
    assert len(dedupe_polys([main_, (line([1.2, 0.0], [1.2, 20.0], 2.0), np.ones(11))])) == 2, "a polyline 1.2 m beside is another road (tolerance 1 m)"
    # the pose error by hand: p' = R(yaw) p + shift (north -> west for 90 degrees), the directions rotate and are not translated; no shift: the yaw alone is not applied; the samples are 2 m apart
    polys = [(line([0.0, 0.0], [0.0, 10.0], 1.0), np.array([1.0] * 5 + [2.0] * 6))]
    pos, d, c = sample_route(polys)
    assert np.allclose(pos[:, 1], np.arange(0.0, 10.01, 2.0)) and np.allclose(d, [0.0, 1.0]) and c.tolist() == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0], "samples every 2 m, direction of travel, counts interpolated and rounded"
    p2, d2, c2 = sample_route(polys, np.array([3.0, 4.0]), np.pi / 2)
    assert np.allclose(p2, np.stack([-pos[:, 1] + 3.0, pos[:, 0] + 4.0], 1), atol=1e-9) and np.allclose(d2, [-1.0, 0.0]) and (c2 == c).all()
    assert np.allclose(sample_route(polys, np.array([3.0, 4.0]), 0.0)[0], pos + [3.0, 4.0]) and np.allclose(sample_route(polys, None, 0.3)[0], pos)
    j = sample_route([(line([0.0, 0.0], [0.0, 6.0]), np.ones(13)), (line([0.0, 6.0], [0.0, 12.0]), np.ones(13))])         # two pieces that join at y = 6: the joint is one sample, not two
    assert len(j[0]) == 7 and np.allclose(j[0][:, 1], [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]), "two pieces that join: the joint is one sample, not two"
    assert len(sample_route([(line([0.0, 0.0], [0.0, 6.0]), np.ones(13)), (line([0.0, 8.0], [0.0, 12.0]), np.ones(9))])[0]) == 4 + 3, "a gap between two pieces is not bridged and no sample is dropped there"
    assert len(sample_route([(np.array([[1.0, 1.0]]), np.ones(1))])[0]) == 0 and len(sample_route([])[0]) == 0 and len(sample_route([(np.zeros((3, 2)), np.ones(3))])[0]) == 0, "a polyline of one point / of no length is left out"
    # the pose error draws: the order of map_sim (direction, length, yaw), the bounds, the generator is the md5 of the token
    r1, r2 = np.random.RandomState(5), np.random.RandomState(5)
    sh, yw = pose_error(r1, 1.0, 2.0)
    ang = r2.uniform(0, 2 * np.pi)
    ln = r2.uniform(1.0, 2.0)
    assert np.allclose(sh, ln * np.array([np.cos(ang), np.sin(ang)])) and abs(yw - np.deg2rad(r2.randn())) < 1e-15 and 1.0 <= np.linalg.norm(sh) <= 2.0
    draws = [pose_error(token_rng(f"t{k}"), *ERROR_SHIFT) for k in range(400)]
    assert all(1.0 <= np.linalg.norm(s_) <= 2.0 for s_, _ in draws) and abs(np.std([np.degrees(y_) for _, y_ in draws]) - 1.0) < 0.15 and abs(np.mean([np.degrees(y_) for _, y_ in draws])) < 0.15
    assert token_rng("a").rand() == token_rng("a").rand() != token_rng("b").rand() and token_rng("a").rand() != token_rng("a", "|gross").rand()
    import map_sim as MS
    assert token_rng("tok1").rand() == MS._rng_for("tok1").rand() and np.allclose(MS.make_map(dict(pieces=[], sample="x"), "error", token="tok1")["shift"], pose_error(token_rng("tok1"), *ERROR_SHIFT)[0]), "the token generator and the pose error are map_sim's"
    assert len({tuple(np.round(pose_error(token_rng(f"q{k}"), 1.0, 2.0)[0], 3)) for k in range(50)}) == 50 and len({np.sign(pose_error(token_rng(f"q{k}"), 1.0, 2.0)[0][1]) for k in range(50)}) == 2, "the direction covers the circle"
    # the error route: degrade_sd (Douglas-Peucker keeps the ends and the corners) after the pose error, the same generator; fixed per token
    rp = [(arc([8.0, 0.0], 8.0, np.pi, np.pi / 2), np.ones(len(arc([8.0, 0.0], 8.0, np.pi, np.pi / 2))))]
    e1, e2, e3 = (error_route(rp, t) for t in ("a", "a", "b"))
    assert all(np.array_equal(x, y) for x, y in zip(e1, e2)) and not np.array_equal(e1[0], e3[0])
    rng = token_rng("a")
    shift, yaw = pose_error(rng, *ERROR_SHIFT)
    want = sample_route(SD.degrade_sd(dict(route=rp, roads=[]), rng)["route"], shift, yaw)
    assert all(np.array_equal(x, y) for x, y in zip(e1, want)), "error route = pose error, then degrade_sd, then the samples, one generator"
    assert max(np.abs(np.linalg.norm(sample_route(rp)[0][:, None] - e1[0][None], axis=2).min(1))) < 6.0 and np.linalg.norm(e1[0][0] - sample_route(rp)[0][0]) > 0.2
    # truncation: the nearest the ego, in the original order; counts clipped to 1 - 5, 0 stays unknown; the table has the fixed keys, one row per sample
    nf = MAX_SAMPLES + 36                                                                       # more samples than the cap
    far_ = (np.stack([np.arange(float(nf)), np.zeros(nf)], 1), np.tile([1.0, 0.0], (nf, 1)), np.full(nf, 7.0))
    t = finish(far_)
    assert len(t["valid"]) == MAX_SAMPLES and np.array_equal(t["pos"][:, 0], np.arange(float(MAX_SAMPLES))) and (t["count"] == COUNT_MAX).all() and sorted(t) == sorted(KEYS) and t["count"].dtype == np.int64 and t["valid"].all()
    assert np.array_equal(finish((far_[0][::-1].copy(), far_[1], far_[2]))["pos"][:, 0], np.arange(MAX_SAMPLES - 1.0, -1.0, -1.0)), "the nearest are kept, in their own order"
    assert (finish(with_count(far_, 0))["count"] == 0).all() and finish((far_[0], far_[1], np.array([0.2] * (nf - 1) + [3.0])))["count"][[0, 1]].tolist() == [1, 1] and finish((far_[0][:3], far_[1][:3], np.array([0.0, 1.0, 3.0])))["count"].tolist() == [0, 1, 3]
    assert len(finish(EMPTY)["valid"]) == 0 and finish(EMPTY)["pos"].shape == (0, 2)
    # the leak check: extra keys of any lane-level kind are refused, missing keys too, a wrong shape too
    good = finish(sample_route(polys))
    for extra in ("piece", "road", "route", "ids", "succ", "lane_index", "on_road", "on_lane", "on_road_id", "label", "pidx"):
        refuses(lambda: check_keys(dict(good, **{extra: np.zeros(len(good["valid"]))})), "a lane-level field is a leak")
    refuses(lambda: check_keys({k: v for k, v in good.items() if k != "count"}), "missing ['count']")
    refuses(lambda: check_keys(dict(good, pos=good["pos"][:, :1])), "route table shapes")
    refuses(lambda: collate_routes([dict(good, road=good["count"])]), "a lane-level field is a leak")
    cb = collate_routes([finish(far_), finish(EMPTY), good])
    assert cb["pos"].shape == (3, MAX_SAMPLES, 2) and cb["valid"].sum(1).tolist() == [MAX_SAMPLES, 0, 6] and not cb["valid"][2, 6:].any() and cb["pos"].dtype == torch.float32 and cb["count"].dtype == torch.int64 and cb["valid"].dtype == torch.bool and (cb["pos"][1] == 0).all()
    ce = collate_routes([finish(EMPTY)] * 2)
    assert ce["pos"].shape == (2, 1, 2) and not ce["valid"].any(), "all absent: one padding column"
    # the draw classes: the boundaries of MIX and the shares over many draws
    assert [draw_class(u) for u in (0.0, 0.2499, 0.25, 0.7199, 0.72, 0.7999, 0.80, 0.8999, 0.90, 0.9999)] == ["exact", "exact", "error", "error", "unknown", "unknown", "absent", "absent", "gross", "gross"]
    cls = [draw_class(np.random.RandomState([0, k, 0]).rand()) for k in range(20000)]
    assert all(abs(cls.count(n_) / len(cls) - p) < 0.012 for n_, p in MIX), {n_: cls.count(n_) / len(cls) for n_, _ in MIX}
    # the wrong plan: (i + 97) mod N, the next frame of another scene (with a route) when that is the same scene; a single scene is refused
    sc = np.repeat(np.arange(5), 40)
    pl = other_scene_plan(sc)
    assert (pl == (np.arange(200) + 97) % 200).all() and (sc[pl] != sc).all()
    assert other_scene_plan([0, 0, 0, 1, 1, 1], None, 1).tolist() == [3, 3, 3, 0, 0, 0] and other_scene_plan([0, 0, 1, 1], None, 1).tolist() == [2, 2, 0, 0]
    assert other_scene_plan([0, 0, 1, 1, 2, 2], [1, 1, 0, 1, 1, 1], 1).tolist() == [3, 3, 4, 4, 0, 0] and other_frame(np.array([0, 1, 1]), np.array([1, 0, 1]), 0, 1) == 2, "a frame without a route is no donor"
    refuses(lambda: other_scene_plan([0, 0, 0]), "one scene only")
    # the cache: pack -> write -> read; the labels padded with -1, the ragged polylines, the exact tokens, the manifest and its guards
    graphs, tokens, scene, recs, arr = toy_world()
    assert arr["on_road"].shape == (12, 24) and arr["n_nodes"].tolist() == [24] * 12 and arr["has_route"].tolist() == [k not in (5, 9) for k in range(12)] and arr["decision"].tolist() == [k not in (5, 9) for k in range(12)]
    with tempfile.TemporaryDirectory() as tmp:
        man = write_cache(os.path.join(tmp, "val"), arr, tokens, dict(split="val", lane_gt="sig"))
        rc = TokenRoute(os.path.join(tmp, "val"))
        assert len(rc) == 12 and man["frames_with_route"] == 10 and man["frames_without_route"] == 2 and man["frames"] == 12 and man["tokens_sha"] == tokens_sha(tokens) and man["version"] == VERSION
        for i in (0, 4, 5, 11):
            on, ln = rc.labels(i)
            assert np.array_equal(on, recs[i]["on_road"]) and np.array_equal(ln, recs[i]["on_lane"]) and np.array_equal(rc.labels_id(i), recs[i]["on_road_id"]) and len(rc.polys(i)) == len(recs[i]["polys"]) and all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(rc.polys(i), recs[i]["polys"]))
            assert all(np.array_equal(rc.exact(i)[k], finish(sample_route(recs[i]["polys"]))[k]) for k in KEYS)
        assert (rc.labels(5)[0] == -1).all() and len(rc.exact(5)["valid"]) == 0 and len(rc.polys(5)) == 0 and rc.a["scene"].tolist() == scene.tolist()
        # a stale or foreign cache is refused: another version, another frame list
        mp = os.path.join(tmp, "val", "manifest.json")
        good_man = json.load(open(mp))
        json.dump(dict(good_man, version="v0"), open(mp, "w"))
        refuses(lambda: TokenRoute(os.path.join(tmp, "val")), "rebuild")
        json.dump(dict(good_man, tokens_sha="x"), open(mp, "w"))
        refuses(lambda: TokenRoute(os.path.join(tmp, "val")), "frame list")
        json.dump(good_man, open(mp, "w"))
        refuses(lambda: TokenRoute(os.path.join(tmp, "nope")), "not a finished route cache")
        json.dump(dict(good_man, params=dict(good_man["params"], max_samples=good_man["params"]["max_samples"] + 1)), open(mp, "w"))
        refuses(lambda: TokenRoute(os.path.join(tmp, "val")), "built with other constants")
        json.dump(good_man, open(mp, "w"))
        assert len(TokenRoute(os.path.join(tmp, "val"), "sig")) == 12 and good_man["params"] == cache_params()
        refuses(lambda: TokenRoute(os.path.join(tmp, "val"), "other_sig"), "built for the lane GT")
        fake = ToyTok([f"toy{k}" for k in range(12)])                                                 # a token cache of another lane GT: RouteData must hand its signature to the reader
        fake.dir = os.path.join(tmp, "tokcache")
        os.makedirs(fake.dir)
        json.dump(dict(lane_gt="other_sig"), open(os.path.join(fake.dir, "meta.json"), "w"))
        refuses(lambda: RouteData(tok=fake, rdir=os.path.join(tmp, "val"), mode="error"), "built for the lane GT")
        json.dump(dict(lane_gt="sig"), open(os.path.join(fake.dir, "meta.json"), "w"))
        assert len(RouteData(tok=fake, rdir=os.path.join(tmp, "val"), mode="error")) == 12 and token_lane_gt(ToyTok(["a"])) is None
        # the dataset on the toy world, every read-out mode by hand
        dd = {m: toy_data(m, rc) for m in MODES}
        assert all("route" in dd[m][0] for m in MODES) and "route" not in toy_data(None, rc)[0], "every mode adds the route key to the item, mode None adds none"
        for i in (0, 4, 10):
            polys_i = rc.polys(i)
            assert all(np.array_equal(dd["exact"].table(i)[k], finish(sample_route(polys_i))[k]) for k in KEYS) and len(dd["exact"].table(i)["valid"]) > 20
            raw_e = error_route(polys_i, f"toy{i}")
            assert all(np.array_equal(x, y) for x, y in zip(dd["error"].raw_route(i), raw_e)), "error = the seeded error route, the same on every call"
            gr = dd["gross"].raw_route(i)
            assert len(gr[0]) == len(raw_e[0]) and np.allclose(np.linalg.norm(gr[0] - raw_e[0], axis=1), 5.0) and np.array_equal(gr[1], raw_e[1]) and np.array_equal(gr[2], raw_e[2]), "gross = the error route shifted rigidly by 5 m"
            assert len(dd["none"].raw_route(i)[0]) == 0
            j = int(dd["wrong"].plan[i])
            assert scene[j] != scene[i] and rc.a["has_route"][j] and all(np.array_equal(x, y) for x, y in zip(dd["wrong"].raw_route(i), error_route(rc.polys(j), f"toy{j}"))), "wrong = the error route of the frame of another scene (the plan)"
        assert len({tuple(dd["gross"].raw_route(i)[0][0] - dd["error"].raw_route(i)[0][0]) for i in range(12) if rc.a["has_route"][i]}) > 6, "the gross direction differs per frame"
        assert len(dd["error"].raw_route(5)[0]) == 0 and len(dd["exact"].raw_route(5)[0]) == 0 and len(dd["wrong"].table(5)["valid"]) > 0, "a frame without a route has no route of its own (the labels are -1)"
        # mode train: the draw counter counts the reads of a frame; each read replays the stream of (init seed, token, draw index): the class, then the pose error, then the degradation; another init seed, another draw
        dt = toy_data("train", rc, init_seed=3)
        a0, a1 = dt.raw_route(2), dt.raw_route(2)
        assert int(dt.draws[2]) == 2 and int(dt.draws[3]) == 0, "the draw counter of a frame counts its reads"
        dt2 = toy_data("train", rc, init_seed=3)
        assert all(np.array_equal(x, y) for x, y in zip(dt2.raw_route(2), a0)) and all(np.array_equal(x, y) for x, y in zip(dt2.raw_route(2), a1)), "the same (init seed, token, draw index) = the same route"
        assert any(not np.array_equal(toy_data("train", rc, init_seed=4).raw_route(2)[k], a0[k]) for k in range(3)) or len(a0[0]) == 0, "another init seed, another draw"
        big, seen, n_read = toy_data("train", rc, init_seed=2), dict.fromkeys([n_ for n_, _ in MIX], 0), 1500
        ex_raw = sample_route(rc.polys(0))
        for k in range(n_read):
            rr = np.random.RandomState([2, zlib.crc32(b"toy0"), k])                                          # the stream of read k of frame 0
            cl = draw_class(rr.rand())
            raw = big.raw_route(0)
            seen[cl] += 1
            if cl == "exact":
                want = ex_raw
            elif cl == "absent":
                want = EMPTY
            elif cl in ("error", "unknown"):
                want = error_like(rc.polys(0), rr, *TRAIN_SHIFT)
                want = with_count(want, 0) if cl == "unknown" else want
            elif rr.rand() < 0.5:
                want = error_like(rc.polys(0), rr, *GROSS_SHIFT)
            else:
                want = error_like(rc.polys(other_frame(scene, arr["has_route"], 0, int(rr.randint(12)))), rr, *TRAIN_SHIFT)
            assert len(raw[0]) == len(want[0]) and all(np.array_equal(x, y) for x, y in zip(raw, want)), f"read {k} ({cl}) is not the replay of its stream"
        assert all(abs(seen[n_] / n_read - p) < 0.04 for n_, p in MIX), seen
        # persistent workers count together (the counter array is in shared memory): two epochs in shuffled order over 2 workers, every read of a frame is the next draw of ITS stream
        ref = toy_data("train", rc, init_seed=2)
        want_t = [[ref.table(i) for i in range(12)] for _ in range(2)]
        dl = torch.utils.data.DataLoader(toy_data("train", rc, init_seed=2), batch_size=3, shuffle=True, generator=torch.Generator().manual_seed(0), num_workers=2, collate_fn=lambda b: b, persistent_workers=True)
        for ep in range(2):
            got = {int(o["token"][3:]): o["route"] for batch in dl for o in batch}
            assert sorted(got) == list(range(12)) and all(np.array_equal(got[i][k], want_t[ep][i][k]) for i in range(12) for k in KEYS), f"epoch {ep}: a read of a frame is not the next draw of its stream (the counters are not shared by the workers)"
        del dl
        # the item and the batch: labels by the token's GT node (token t of frame i -> table node 5 t mod 24, -1 for t >= 16), -1 without a route, the underlying item keys untouched, mode None = no route key
        d0 = toy_data("error", rc)
        it = d0[0]
        assert "route" in it and "route" not in toy_data(None, rc)[0], "mode None adds no route key, every other mode does"
        gt = ToyTok(["x"])[0]["label"]["gt_id"]
        on_r, on_l = rc.labels(0)
        on_i = rc.labels_id(0)
        assert np.array_equal(it["on_road_id"], np.where(gt >= 0, on_i[np.clip(gt, 0, None)], -1)) and not np.array_equal(on_i, on_r), "on_road_id is its own array"
        want_r = np.where(gt >= 0, on_r[np.clip(gt, 0, None)], -1).astype(np.int8)
        assert np.array_equal(it["on_road"], want_r) and np.array_equal(it["on_lane"], np.where(gt >= 0, on_l[np.clip(gt, 0, None)], -1)) and it["on_road"].dtype == np.int8 and (it["on_road"][16:] == -1).all() and (it["on_road"][:16] >= 0).all()
        assert (d0[5]["on_road"] == -1).all() and (d0[5]["on_lane"] == -1).all() and not d0[5]["has_route"] and len(d0[5]["route"]["valid"]) == 0 and d0[0]["has_route"] and d0[0]["decision"]
        assert sorted(it) == sorted(list(ToyTok(["x"])[0]) + ["on_road", "on_lane", "on_road_id", "has_route", "decision", "route"]) and sorted(toy_data(None, rc)[0]) == sorted(list(ToyTok(["x"])[0]) + ["on_road", "on_lane", "on_road_id", "has_route", "decision"]), "the item keys of the loader under it are untouched"
        for lab in ("on_road", "on_lane", "on_road_id"):                                                              # a token with a GT node but a node that is off the route is 0, on it 1, no node -1
            assert set(np.unique(it[lab][:16])) <= {0, 1}
        refuses(lambda: toy_data("nope", rc), "nope")
        refuses(lambda: toy_data("error", rc).raw_route(0, "wrong"), "needs the scene plan")
        refuses(lambda: toy_data("error", rc).raw_route(0, "train"), "mode train")
        refuses(lambda: RouteData(tok=ToyTok(["a", "b"]), rc=rc, mode="error"), "not the frame list")
    toy_loader_test()
    print("lane_route_data selftest ok")


def toy_loader_test():
    """the batch of the loader on a toy cache: the keys of `lane_ptok_data.collate` untouched, the route padded, the labels stacked"""
    graphs, tokens, scene, recs, arr = toy_world()
    with tempfile.TemporaryDirectory() as tmp:
        write_cache(os.path.join(tmp, "val"), arr, tokens, dict(split="val", lane_gt="sig"))
        rc = TokenRoute(os.path.join(tmp, "val"))
        real = LP.collate
        try:
            LP.collate = lambda b: dict(token=[o["token"] for o in b], row=torch.tensor([o["row"] for o in b]), marker=torch.ones(1))                 # the underlying batch of the stand-in
            ds = toy_data("error", rc)
            b = collate([ds[0], ds[5], ds[1]])
            assert sorted(b) == sorted(["token", "row", "marker", "on_road", "on_lane", "on_road_id", "has_route", "decision", "route"]) and b["marker"].tolist() == [1.0] and b["token"] == ["toy0", "toy5", "toy1"], "the underlying batch keys are untouched"
            assert b["on_road"].shape == (3, 20) and b["on_road"].dtype == torch.int8 and (b["on_road"][1] == -1).all() and b["has_route"].tolist() == [True, False, True] and b["decision"].dtype == torch.bool
            assert sorted(b["route"]) == sorted(KEYS) and b["route"]["pos"].shape[0] == 3 and b["route"]["valid"][1].sum() == 0 and b["route"]["valid"][0].sum() > 10
            assert "route" not in collate([toy_data(None, rc)[0], toy_data(None, rc)[1]]) and toy_data("none", rc)[0]["route"]["valid"].shape == (0,)
        finally:
            LP.collate = real


# ------------------------------------------------------------------------------------------------------------------------------ command line
def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("build")
    s.add_argument("--split", required=True, choices=("val", "train"))
    s.add_argument("--limit", type=int)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--ident", default="e292_a")
    s = sub.add_parser("loadercheck")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--ident", default="e292_a")
    s = sub.add_parser("check")
    s.add_argument("--n", type=int)
    s.add_argument("--split", default="both", choices=("val", "train", "both"))
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--ident", default="e292_a")
    s.add_argument("--out")
    for name in ("fixture", "stats"):
        s = sub.add_parser(name)
        s.add_argument("--split", default="val", choices=("val", "train"))
        s.add_argument("--n", type=int, default=200)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--ident", default="e292_a")
        s.add_argument("--out")
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "build":
        return build(a.split, a.seed, a.ident, a.limit)
    if a.cmd == "loadercheck":
        return loader_check(a.seed, a.ident)
    res = check(a.n, a.seed, a.ident, splits=("val", "train") if a.split == "both" else (a.split,)) if a.cmd == "check" else (fixture(a.split, a.n, a.seed, a.ident) if a.cmd == "fixture" else stats(a.split, a.seed, a.ident))
    if a.out:
        p = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
        with open(p, "w") as fh:
            json.dump(res, fh, indent=1, default=str)
    if a.cmd == "fixture":
        assert res["passed"], "the fixture FAILED"


if __name__ == "__main__":
    main()
