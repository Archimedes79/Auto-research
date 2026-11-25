"""Read-out, scoring and reference rows of E389 (A6, D127: the route-association head `net/model/jroute.py` on the cached B' tokens of the champion); the data is `net/prep/lane_route_data.py`, the card is
`experiments/E389_sd-route-association-of-champion-tokens/experiment.md` (section 0, the rule and the read-outs).

    wsl bash tools/env/_py14.sh jroute_eval.py dump  --run R --seed S --mode exact|error|none|wrong|gross [--cpu] [--bs 8] [--workers 0] [--max N]          # the head on the VAL frames -> runs/R/route_<mode>[_n<N>].pkl (never overwritten)
    wsl bash tools/env/_py14.sh jroute_eval.py score --run R --seed S --mode error [--ref-run Z [--ref-mode error]] [--champion RUN] [--keep-thr 0.6] [--draws 1000] [--max N]   # CPU -> runs/R/route_card_<mode>[_vs_<Z>[_rm<M>]][_n<N>].json
    wsl bash tools/env/_py14.sh jroute_eval.py rules --seed S --split val [--modes exact,error,gross,wrong] [--champion RUN] [--max N]                       # CPU, no head: flag-all, straight prior, geometric rule -> runs/e389_rules/rules_card_val_s<S>[_n<N>].json
    wsl bash tools/env/_py14.sh jroute_eval.py cost  --run R [--K 512] [--M cap]                                                                                # CPU: parameters and GMACs of one frame, NOT timed -> runs/R/route_cost.json (the card's clause 6 when no quiet GPU window exists)
    wsl bash tools/env/_py14.sh jroute_eval.py selftest                                                                                                      # hand cases and bootstrap pairing, no data; mutants: tests/test_jroute_eval.py

Unit = the COVERED GT NODE, F1 per node. A GT node of a frame's E357 table is covered when a DELIVERED token is assigned to it (the cache's `gt_id`; one token per node at most); its label is the GT label of the node (the route cache's on_road / on_lane,
the labels of `lane_route_data`), its prediction is p >= 0.5 of that token. Delivered tokens, two definitions, both always reported (`by_delivered`): `kept` (THE ONE USED: the champion's stage-1 head at the D115 decode = the valid tokens of the E373 dump
`runs/e373_cb_s<S>/token_val.pkl` with p_none < 0.6; the dump's own `keep` is the record decode 0.5) and `proxy` (the card critic's proxy: valid tokens with the cached existence e_p >= 0.5). Only frames that have a route are scored (all 873 of VAL).
route_F1_road / route_F1_lane = pooled F1 over the covered nodes of those frames (P, R beside); fork frames = the data layer's `decision` frames (a route piece with more than one successor ends within 40 m ahead; merges are not counted), F1_road on
them pooled; the card needs >= 60 fork frames from >= 8 scenes (`fork.clause`). Route success@30 m per frame: >= 3 covered ROUTE nodes (on_road) within 30 m ahead (the token's position: y >= 0 and |pos| <= 30 m; the vehicle frame has forward = +y)
-> the frame is ELIGIBLE; it SUCCEEDS when >= 90 % of those are flagged and <= 20 % of the covered non-route nodes within 30 m ahead are flagged; success30 = successes / eligible frames, success30_all = successes / frames with a route.
False tokens = delivered valid tokens without a GT node (gt_id -1, unscored): their flag rate is reported. Calibration = per bin of p of 0.1 over the covered nodes the share that is on the route (road, lane). Paired scene bootstrap (the repository's
`Evaluation/stats.py`, scenes of the route cache, --draws 1000, seed 0, a frame drawn twice counts twice) against --ref-run over the same frames: route_F1_road, route_F1_lane, fork_F1_road, success30; the card also holds the interval of the system
alone. Arm z (the matched no-map reference) is dumped in any mode (its route is forced invalid): the paired comparisons of the trust clause use --ref-mode (e.g. m in mode none / wrong against z in mode error). A pair is refused unless both dumps hold the
same frames, valid tokens and route cache (version, lane GT, frame list, route cap). A dump is refused when its model.pt is not the one that made it, and `dump` refuses a head that was not trained on the route cache it is read on (`check_trained_on`: a route cache recorded in the run's args must equal the cache's; a model.pt without that record must not be older than the cache's manifest: the trainer does not write the record yet). A dump, a card and a rules card are never overwritten. With --ref-run and a rules card of this seed on this very route cache the card also holds `gap_closed` = (m - z) / (rule - z) of the geometric rule of the mode (exact / error).
Reference rows (`rules`, CPU, 'rule', never a gate): flag-all; the no-map STRAIGHT PRIOR (|x| <= 3.5 m and cos(heading, +y) >= 0.7); the GEOMETRIC RULE: a token is on the route when it lies within n * 1.75 m + 0.5 m of the route polyline (n = the lane count of
the nearest route segment's nearer sample, 0 = 1) and the heading cosine to that segment's direction is >= 0.7, on the route tokens of the mode error (the data layer's fixed draw per frame), the exact ones, and gross (error + 5 m) and wrong (the route of another scene: what the rule does with a wrong map). The rules flag the road level; their lane
numbers read the same flags against on_lane and are no lane rule. The card's critic numbers (EXPECT) were measured on other labels; `reproduced` says whether this data layer's labels give them within 0.01.
Tests: evaluation/tests/test_jroute_eval.py (selftest + textual mutants + a dump / score / rules run on the first frames of the real VAL cache with a fresh head).
"""
import argparse
import hashlib
import itertools
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub_ in ("evaluation", "training", "net/prep", "net/model", "net/loss", "training/data", "training/data/prep", "training/train", "../autoresearch"):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
from Evaluation import stats as ST  # noqa: E402  (the shared scene bootstrap)

RUNS = os.path.join(ROOT, "runs")
THR = 0.5                                                                   # a token is flagged at p >= THR
KEEP_THR = 0.6                                                              # the D115 decode of the champion: a token is kept at p_none < 0.6
PROXY_E = 0.5                                                               # the critic's proxy: delivered = the cached existence probability >= 0.5
SUCC_M, SUCC_MIN, SUCC_FLAG, SUCC_FALSE = 30.0, 3, 0.9, 0.2                 # route success@30 m: >= 3 covered route nodes within 30 m ahead, >= 90 % flagged, <= 20 % of the non-route nodes flagged
STRAIGHT_X, COS_MIN = 3.5, 0.7                                              # the straight prior: |x| <= 3.5 m and cos(heading, +y) >= 0.7; the geometric rule's heading cosine
HALF_LANE, MARGIN, GAP = 1.75, 0.5, 2.5                                     # the geometric rule: n * 1.75 + 0.5 m from the polyline; two route samples further apart than GAP m are two polylines
BINS = 10                                                                   # the calibration bins of p
MIN_FORK_FRAMES, MIN_FORK_SCENES = 60, 8                                    # the card's minimum for the fork clauses
MODES = ("exact", "error", "none", "wrong", "gross")
EXPECT = dict(flag_all=0.490, straight_prior=0.778, geometric_error=0.959, geometric_exact=1.000)      # the card critic's F1_road on the proxy-delivered nodes of 788 VAL frames
TOL = 0.01
DRAWS = 1000
BOOT_KEYS = ("route_F1_road", "route_F1_lane", "fork_F1_road", "success30")
COLS = ("frames", "fork", "nodes", "tp_r", "fp_r", "fn_r", "tp_l", "fp_l", "fn_l", "elig", "succ", "false", "false_r", "false_l")        # the per-frame count columns: everything the pooled numbers and the bootstrap sum
IX = {c: i for i, c in enumerate(COLS)}


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def _file_id(path):
    """sha256 and mtime of a file (the identity of a stored model or dump)"""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return dict(path=os.path.relpath(path, ROOT) if path.startswith(ROOT) else path, sha256=h.hexdigest(), mtime=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(os.path.getmtime(path))) + " UTC")


def _default(x):
    return x.item() if isinstance(x, np.generic) else x.tolist() if isinstance(x, np.ndarray) else str(x)


def write_new(path, obj):
    """a card of json, never overwritten"""
    assert not os.path.exists(path), f"a stored card is never overwritten: {path}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "x") as fh:
        json.dump(obj, fh, indent=1, default=_default)
    return path


# ------------------------------------------------------------------------------------------------------------------------------ the numbers
def div(a, b):
    return float(a) / float(b) if b else float("nan")


def f1(tp, fp, fn):
    return div(2 * tp, 2 * tp + fp + fn)


def flags(p):
    """the system's flag: p >= THR (a NaN, the padding token's probability, is not flagged)"""
    return np.asarray(p) >= THR


def delivered_kept(valid, pnone, keep_thr=KEEP_THR):
    """the tokens of the champion's stage-1 head at the D115 decode: valid and p_none < keep_thr (float32 compare, as `jtoken_eval.decode_variant`)"""
    return np.asarray(valid, bool) & (np.asarray(pnone, np.float32) < np.float32(keep_thr))


def delivered_proxy(valid, e_p, thr=PROXY_E):
    """the critic's proxy: valid tokens whose cached existence probability is >= thr"""
    return np.asarray(valid, bool) & (np.asarray(e_p) >= thr)


def count_matrix(side, delivered, road, lane):
    """(F, len(COLS)) float: per frame the counts of everything the pooled numbers need. side = the arrays of `load_side` (F frames, K tokens), delivered (F, K) bool, road / lane (F, K) bool = the flags of the system (p >= THR).
    A COVERED node is a delivered valid token with a GT node (gt_id >= 0) in a frame that has a route; padding tokens, undelivered tokens and frames without a route are never scored; a delivered valid token without a GT node is a FALSE token (counted
    apart); route nodes = on_road, success@30 m as in the module docstring"""
    has = side["has_route"][:, None]
    cov = delivered & side["valid"] & (side["gt_id"] >= 0) & has
    assert ((side["lab_road"][cov] >= 0) & (side["lab_lane"][cov] >= 0)).all(), "a covered node without a label"
    yr, yl = side["lab_road"] == 1, side["lab_lane"] == 1
    n = lambda m: m.sum(1)  # noqa: E731
    near = cov & (side["pos"][..., 1] >= 0) & (np.linalg.norm(side["pos"], axis=-1) <= SUCC_M)
    n_route, n_flag, n_non, n_non_flag = n(near & yr), n(near & yr & road), n(near & ~yr), n(near & ~yr & road)
    elig = n_route >= SUCC_MIN
    succ = elig & (n_flag >= SUCC_FLAG * n_route - 1e-9) & (n_non_flag <= SUCC_FALSE * n_non + 1e-9)
    fal = delivered & side["valid"] & (side["gt_id"] < 0) & has
    col = dict(frames=has[:, 0], fork=side["decision"] & side["has_route"], nodes=n(cov), tp_r=n(cov & yr & road), fp_r=n(cov & ~yr & road), fn_r=n(cov & yr & ~road), tp_l=n(cov & yl & lane), fp_l=n(cov & ~yl & lane),
               fn_l=n(cov & yl & ~lane), elig=elig, succ=succ, false=n(fal), false_r=n(fal & road), false_l=n(fal & lane))
    return np.stack([np.asarray(col[c], np.float64) for c in COLS], 1)


def numbers(C, sel=None):
    """the pooled numbers of the frames `sel` (indices, a frame twice counts twice; None = all) of a count matrix: route_F1 / P / R at the road and the lane level, the same for the fork frames at the road level, route success, the false tokens' flag rate"""
    C = C if sel is None else C[sel]
    s, fk = (dict(zip(COLS, m.sum(0))) for m in (C, C[C[:, IX["fork"]] == 1]))
    out = dict(frames=int(s["frames"]), nodes=int(s["nodes"]), fork_frames=int(fk["frames"]), eligible_frames=int(s["elig"]), success_frames=int(s["succ"]))
    for lvl, k in (("road", "r"), ("lane", "l")):
        out.update({f"route_F1_{lvl}": f1(s[f"tp_{k}"], s[f"fp_{k}"], s[f"fn_{k}"]), f"route_P_{lvl}": div(s[f"tp_{k}"], s[f"tp_{k}"] + s[f"fp_{k}"]), f"route_R_{lvl}": div(s[f"tp_{k}"], s[f"tp_{k}"] + s[f"fn_{k}"])})
    out.update(fork_F1_road=f1(fk["tp_r"], fk["fp_r"], fk["fn_r"]), fork_P_road=div(fk["tp_r"], fk["tp_r"] + fk["fp_r"]), fork_R_road=div(fk["tp_r"], fk["tp_r"] + fk["fn_r"]), success30=div(s["succ"], s["elig"]),
               success30_all=div(s["succ"], s["frames"]), false_tokens=int(s["false"]), false_flag_road=div(s["false_r"], s["false"]), false_flag_lane=div(s["false_l"], s["false"]))
    return out


def calibration(side, delivered, p, label):
    """the share of covered nodes that are on the route by bin of p (BINS bins of 0.1, p = 1 in the last): rows of lo, hi, n, share_on, mean_p; `label` = side['lab_road'] or side['lab_lane']. p is read at the covered nodes only"""
    cov = delivered & side["valid"] & (side["gt_id"] >= 0) & side["has_route"][:, None]
    q, y = np.asarray(p, np.float64)[cov], label[cov]
    assert np.isfinite(q).all(), "a covered node without a probability"
    b = np.minimum((q * BINS).astype(int), BINS - 1)
    return [dict(lo=k / BINS, hi=(k + 1) / BINS, n=int((b == k).sum()), share_on=div((y[b == k] == 1).sum(), (b == k).sum()), mean_p=float(q[b == k].mean()) if (b == k).any() else float("nan")) for k in range(BINS)]


def fork_info(side):
    """the fork frames of the scored frames and the scenes they lie in, and the card's minimum (>= 60 frames from >= 8 scenes): the clause is void below it"""
    fk = side["decision"] & side["has_route"]
    nf, ns = int(fk.sum()), len(set(side["scene"][fk].tolist()))
    ok = nf >= MIN_FORK_FRAMES and ns >= MIN_FORK_SCENES
    return dict(fork_frames=nf, fork_scenes=ns, minimum=[MIN_FORK_FRAMES, MIN_FORK_SCENES], clause="ok" if ok else f"void: {nf} fork frames from {ns} scenes, the card needs >= {MIN_FORK_FRAMES} from >= {MIN_FORK_SCENES}")


def draws_of(scene, B=DRAWS):
    """the frame indices of every scene-bootstrap draw (the repository's stream, seed 0), as a list: a pair and a single system use the same draws"""
    return list(ST.scene_draws(np.asarray(scene), B, 0))


def interval(C, draws, keys=BOOT_KEYS):
    """one system: value and 95 % interval over the scene draws of each key"""
    return ST.interval_draws(lambda sel: numbers(C, sel), draws, keys)


def paired(Ca, Cb, draws, keys=BOOT_KEYS):
    """a - b (a = the variant, b = the control) of each key, pooled per draw, with the 95 % interval over the SAME scene draws"""
    return ST.paired_draws(lambda sel: numbers(Ca, sel), lambda sel: numbers(Cb, sel), draws, keys)


# ------------------------------------------------------------------------------------------------------------------------------ the data of a split
def load_side(seed, ident="e292_a", K=None, cache=None, n_max=0, split="val", mode=None):
    """everything the scorer reads besides the system, from the cache of the cached B' tokens and the route cache: tokens, scene, has_route, decision (F,), gt_id, valid, e_p (F, K), pos, dirv (F, K, 2), lab_road, lab_lane (F, K) int8 (the label of the
    token's GT node, -1 none) and `ds` (the `lane_route_data.RouteData`; `mode` = its map mode for `ds.table(i, mode)`: only the mode wrong needs it at construction, for the scene plan; it checks the TEST guard and that the two caches hold the same frames). Only VAL"""
    import lane_route_data as LRD
    assert split == "val", f"split {split!r}: E389 reads VAL only"
    ds = LRD.RouteData(split, seed, ident, K, mode, **({"cache": cache} if cache else {}))
    n = n_max or len(ds)
    f = ds.tok.c.f
    side = dict(tokens=list(ds.tokens[:n]), scene=np.asarray(ds.scene[:n]), has_route=np.asarray(ds.has[:n], bool), decision=np.asarray(ds.rc.a["decision"][:n], bool), ds=ds,
                **{k: np.asarray(f[k][:n]) for k in ("gt_id", "valid", "e_p", "pos", "dirv")})
    side["valid"] = side["valid"].astype(bool)
    got = side["gt_id"] >= 0
    for key, name in (("lab_road", "on_road"), ("lab_lane", "on_lane")):                  # the label of a token's GT node, read from the route cache's arrays by name; -1 = no node (a frame without a route is -1 throughout)
        lab = np.take_along_axis(np.asarray(ds.rc.a[name][:n]), np.maximum(side["gt_id"], 0), 1)
        side[key] = np.where(got, lab, -1).astype(np.int8)
    return side


def champion_pnone(side, seed, run=None, runs=RUNS):
    """-> (p_none (F, K) float32, the dump's identity): the E373 dump of the champion's stage-1 head on the same frames; refused unless its tokens are the frames' and its `keep` (the record decode, p_none < 0.5) is the valid tokens at p_none < 0.5"""
    path = os.path.join(runs, run or f"e373_cb_s{seed}", "token_val.pkl")
    d = pickle.load(open(path, "rb"))
    fr, n = d["frames"], len(side["tokens"])
    assert [f["token"] for f in fr[:n]] == side["tokens"], f"{path} does not hold the frames of the cache, in order"
    pn = np.stack([np.asarray(f["pnone"], np.float32) for f in fr[:n]])
    assert np.array_equal(delivered_kept(side["valid"], pn, 0.5), np.stack([np.asarray(f["keep"], bool) for f in fr[:n]])), f"{path}: its kept tokens are not the valid tokens at p_none < 0.5"
    return pn, _file_id(path)


IDENT_KEYS = ("version", "lane_gt", "tokens_sha", "max_samples")           # a dump and a card belong to ONE route cache: all four
TRAINED_KEYS = ("version", "lane_gt", "max_samples")                       # a head trained on the TRAIN cache is read on the VAL cache: the label definition, the GT and the route cap must be the same, the frame list (tokens_sha) differs by design


def route_cache_identity(man):
    """what a route cache is (its manifest): the version, the lane GT, the frame list and the route cap (samples per frame) - a head, a dump and a card belong to ONE of these"""
    return dict(version=man["version"], lane_gt=man.get("lane_gt"), tokens_sha=man["tokens_sha"], max_samples=man["params"]["max_samples"])


def check_trained_on(args, model_mtime, manifest, manifest_mtime, name="model.pt"):
    """a head is read only on the route cache version it was trained on (the route cap is part of it: a head trained on a cache that cut the route at another length has seen other tables). When the run's args hold `route_cache` (the trainer writes the
    identity of its TRAIN cache) its version, lane GT and route cap (TRAINED_KEYS; not the frame list: TRAIN and VAL differ) must equal those of the cache being read; a run folder without that record (older runs, fresh heads) must not be older than the cache's manifest
    (a head older than the cache cannot have been trained on it)"""
    now, got = route_cache_identity(manifest), args.get("route_cache")
    if got is not None:
        assert all(got.get(k) == now[k] for k in TRAINED_KEYS), f"{name} was trained on the route cache {got}, the cache being read is {now}: a head is read on the route cache version it was trained on (version, lane GT, route cap {now['max_samples']})"
    else:
        assert model_mtime >= manifest_mtime, (f"{name} is older than the route cache (route cap {now['max_samples']}, version {now['version']}) and records no route cache in its args: it cannot have been trained on it; "
                                               f"retrain the head on this cache (a head trained before the rebuild is read only on the cache it was trained on)")


def cache_id(side):
    """the identity of the data a card was made on: the token cache directory and the route cache (`route_cache_identity`), so that a card made on an older label definition or another route cap is recognised"""
    return dict(token_cache=os.path.relpath(side["ds"].tok.dir, ROOT), route_cache=dict(dir=os.path.relpath(side["ds"].rc.dir, ROOT), **route_cache_identity(side["ds"].rc.manifest)))


def delivered_sets(side, pnone, keep_thr=KEEP_THR):
    return dict(kept=delivered_kept(side["valid"], pnone, keep_thr), proxy=delivered_proxy(side["valid"], side["e_p"]))


# ------------------------------------------------------------------------------------------------------------------------------ the reference rules
def straight_flags(side):
    """the no-map straight prior: |x| <= 3.5 m and cos(heading, +y) >= 0.7"""
    return (np.abs(side["pos"][..., 0]) <= STRAIGHT_X) & (side["dirv"][..., 1] >= COS_MIN)


def route_segments(route):
    """the segments of a route table (pos, dir, count, valid; samples in route order): the pairs of neighbouring samples no further apart than GAP m, then every sample once as a point segment (an isolated sample, a one-sample route) -> a, b (S, 2), unit
    direction (S, 2), count of the nearer end (S, 2) = (count at a, count at b)"""
    v = np.asarray(route["valid"], bool)
    pos, dr, cnt = np.asarray(route["pos"], np.float64)[v], np.asarray(route["dir"], np.float64)[v], np.asarray(route["count"])[v]
    conn = np.linalg.norm(pos[1:] - pos[:-1], axis=1) <= GAP
    a, b = np.concatenate([pos[:-1][conn], pos]), np.concatenate([pos[1:][conn], pos])
    ab = b - a
    ln = np.linalg.norm(ab, axis=1, keepdims=True)
    d = np.where(ln > 1e-9, ab / np.maximum(ln, 1e-9), np.concatenate([dr[:-1][conn], dr]))
    return a, b, d, np.stack([np.concatenate([cnt[:-1][conn], cnt]), np.concatenate([cnt[1:][conn], cnt])], 1)


def geometric_flags(pos, dirv, route):
    """(K,) bool: the geometric rule of one frame: the token's nearest route segment is within n * 1.75 m + 0.5 m (n = the lane count of the segment's nearer end, 0 counts as 1) and the token's heading has a cosine >= 0.7 to that segment's direction.
    No route sample: nothing is on the route"""
    if not np.asarray(route["valid"]).any():
        return np.zeros(len(pos), bool)
    a, b, d, cnt = route_segments(route)
    ab = b - a
    t = np.clip(((pos[:, None, :] - a[None]) * ab[None]).sum(-1) / np.maximum((ab ** 2).sum(-1), 1e-12)[None], 0.0, 1.0)         # (K, S) the foot on each segment
    dist = np.linalg.norm(pos[:, None, :] - (a[None] + t[..., None] * ab[None]), axis=-1)
    j = dist.argmin(1)
    k = np.arange(len(pos))
    n = np.maximum(np.where(t[k, j] < 0.5, cnt[j, 0], cnt[j, 1]), 1)
    return (dist[k, j] <= n * HALF_LANE + MARGIN) & ((dirv * d[j]).sum(-1) >= COS_MIN)


def rule_rows(side, modes, delivered):
    """-> {row: flags (F, K) bool} of the reference rows: flag_all, straight_prior, geometric_<mode> for each mode (route tokens of `lane_route_data.RouteData.table`: exact = the exact tokens, error = the fixed draw per frame)"""
    rows = dict(flag_all=np.ones(side["valid"].shape, bool), straight_prior=straight_flags(side))
    for mode in modes:
        rows[f"geometric_{mode}"] = np.stack([geometric_flags(side["pos"][i].astype(np.float64), side["dirv"][i].astype(np.float64), side["ds"].table(i, mode)) for i in range(len(side["tokens"]))])
    return rows


# ------------------------------------------------------------------------------------------------------------------------------ dump
def dump_frames(p_road, p_lane, valid, tokens, rows):
    """the frames of a dump from the probabilities (B, K) of a batch: token, row, valid, p_road, p_lane (float32; a padding token has NaN: it is never a prediction)"""
    valid = np.asarray(valid, bool)
    return [dict(token=tokens[i], row=int(rows[i]), valid=valid[i].copy(), p_road=np.where(valid[i], p_road[i], np.nan).astype(np.float32), p_lane=np.where(valid[i], p_lane[i], np.nan).astype(np.float32)) for i in range(len(tokens))]


def n_sfx(n_max):
    return f"_n{n_max}" if n_max else ""


def dump_path(runs, run, mode, n_max=0):
    return os.path.join(runs, run, f"route_{mode}{n_sfx(n_max)}.pkl")


def dump(a, runs=RUNS):
    """the head of runs/R/model.pt (eval mode, no_grad) on the VAL frames in the cache order with the map mode `a.mode` of `lane_route_data.loader`; an arm z head reads the route forced invalid (`jroute_train.strip_route`)"""
    import jroute as JR
    import jroute_train as JRT
    import lane_route_data as LRD
    assert a.mode in MODES, f"--mode {a.mode}: {MODES}"
    path = dump_path(runs, a.run, a.mode, a.max)
    assert not os.path.exists(path), f"a stored dump is never overwritten: {path}"
    dev = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    if dev == "cuda":
        import gpu_gate
        gpu_gate.wait_for_room(who=f"jroute_eval dump {a.run}")
    m, args = JRT.load_model(a.run, dev, runs)
    assert args["seed"] == a.seed, f"--seed {a.seed} does not match the run (cache seed {args['seed']})"
    ident, K, cache = args.get("ident") or "e292_a", args.get("K"), args.get("cache")
    ds, dl = LRD.loader("val", a.seed, a.bs, a.workers, mode=a.mode, ident=ident, K=K, shuffle=False, **({"cache": cache} if cache else {}))
    check_trained_on(args, os.path.getmtime(os.path.join(runs, a.run, "model.pt")), ds.rc.manifest, os.path.getmtime(os.path.join(ds.rc.dir, "manifest.json")), f"runs/{a.run}/model.pt")
    n = a.max or len(ds)
    frames = []
    for k, b in enumerate(itertools.islice(dl, -(-n // a.bs))):
        x = JRT.to_device(b, args.get("inputs") or "full", dev)
        if k == 0:
            JR.check_batch(x)
        if args.get("arm") == "z":
            x = JRT.strip_route(x)
        with torch.no_grad():
            out = m(x)
        frames += dump_frames(*(torch.sigmoid(out[s]).float().cpu().numpy() for s in ("road", "lane")), b["valid"].numpy(), b["token"], b["row"].tolist())
    frames = frames[:n]
    meta = dict(run=a.run, seed=a.seed, mode=a.mode, split="val", arm=args.get("arm"), inputs=args.get("inputs") or "full", attn=args.get("attn") or "ab", frames=len(frames), tokens=[f["token"] for f in frames], K=ds.tok.K, ident=ident, cache_root=cache, cache=ds.tok.dir,
                route_cache=dict(dir=os.path.relpath(ds.rc.dir, ROOT), **route_cache_identity(ds.rc.manifest)), model=_file_id(os.path.join(runs, a.run, "model.pt")), steps=args.get("steps"),
                device=dev, autocast=False, torch=torch.__version__, max=a.max)             # fp32, no autocast: the training ran bf16 autocast on the GPU (`jroute_train`)
    with open(path, "wb") as fh:
        pickle.dump(dict(meta=meta, frames=frames), fh)
    print(f"written {os.path.relpath(path, ROOT) if path.startswith(ROOT) else path}: {len(frames)} frames, K {ds.tok.K}, arm {args.get('arm')}, mode {a.mode}, device {dev}")
    return path


# ------------------------------------------------------------------------------------------------------------------------------ score
def read_dump(runs, run, seed, mode, n_max=0):
    """a dump of this read-out -> dict(meta, path, p_road, p_lane (F, K) float32, valid, tokens); refused unless it is the dump of runs/R/model.pt as it is now and of this route cache"""
    path = dump_path(runs, run, mode, n_max)
    d = pickle.load(open(path, "rb"))
    meta, fr = d["meta"], d["frames"]
    assert meta["run"] == run and meta["seed"] == seed and meta["mode"] == mode and meta["split"] == "val" and meta["max"] == n_max and [f["token"] for f in fr] == meta["tokens"], f"{path} is not the dump of this read-out (run, seed, mode, frames)"
    assert meta["model"]["sha256"] == _file_id(os.path.join(runs, run, "model.pt"))["sha256"], f"{path}: runs/{run}/model.pt is not the model that made this dump"
    return dict(meta=meta, path=path, tokens=meta["tokens"], valid=np.stack([f["valid"] for f in fr]), p_road=np.stack([f["p_road"] for f in fr]), p_lane=np.stack([f["p_lane"] for f in fr]))


def check_route_cache(made, now, what):
    """a dump was made on the route cache that is read now: the version, the lane GT, the frame list and the route cap are those of the cache"""
    assert all(made[k] == now[k] for k in IDENT_KEYS), f"{what} was made on another route cache ({made} against {now})"


def check_pair(d, r, side):
    """two dumps of one pair, and the dump and the cache: the same frames, the same tokens, the same route cache (each of the two dumps carries the identity of the route cache it was made on)"""
    assert d["tokens"] == side["tokens"], "the dump does not hold the frames of the cache, in order"
    assert r is None or (r["tokens"] == d["tokens"] and np.array_equal(r["valid"], d["valid"]) and r["meta"]["seed"] == d["meta"]["seed"] and r["meta"]["cache"] == d["meta"]["cache"]), "the reference dump holds other frames or tokens (another cache)"
    assert np.array_equal(d["valid"], side["valid"]), "the dump's valid tokens are not the cache's"
    now = cache_id(side)["route_cache"]
    check_route_cache(d["meta"]["route_cache"], now, "the dump")
    if r is not None:
        check_route_cache(r["meta"]["route_cache"], now, "the reference dump")


def system_numbers(side, delivered, d, draws):
    """one system under one delivered definition -> (its count matrix, the block of the card: numbers, calibration, fork info, interval)"""
    for p in (d["p_road"], d["p_lane"]):
        assert np.isfinite(p[delivered & side["valid"]]).all(), "a delivered valid token without a probability"
    C = count_matrix(side, delivered, flags(d["p_road"]), flags(d["p_lane"]))
    return C, dict(numbers=numbers(C), delivered_tokens_per_frame=float((delivered & side["valid"])[side["has_route"]].sum(1).mean()), calibration=dict(road=calibration(side, delivered, d["p_road"], side["lab_road"]),
                   lane=calibration(side, delivered, d["p_lane"], side["lab_lane"])), fork=fork_info(side), interval=interval(C, draws))


DEFINITIONS = dict(unit="the covered GT node (a delivered token is assigned to it); F1 per node, pooled over the frames that have a route", delivered_used="kept", kept="valid tokens with p_none < keep_thr of the champion's dump (the D115 decode)",
                   proxy="valid tokens with the cached existence e_p >= 0.5 (the card critic's proxy)", flag=f"p >= {THR}", success30=f"eligible = >= {SUCC_MIN} covered route nodes within {SUCC_M:g} m ahead (y >= 0); success = >= {SUCC_FLAG:g} of them flagged and <= {SUCC_FALSE:g} of the covered non-route nodes there flagged",
                   fork="the data layer's decision frames: a route piece with more than one successor ends within 40 m ahead")


RULE_ROW = dict(exact="geometric_exact", error="geometric_error")           # the geometric rule that a map head is measured against in a mode (no row for the other modes)


def gap_closed(m, z, rule):
    """the share of the gap between the no-map head z and the geometric rule that the map head m closes: (m - z) / (rule - z); all F1_road of one delivered definition"""
    return div(m - z, rule - z)


def rule_reference(runs, side, seed, mode, keep_thr, n_max=0):
    """-> {delivered definition: F1_road of the geometric rule of `mode`} from the rules card of this seed, or None: no rule for the mode, no card, or a card made on another route cache (`route_cache_identity`) or at another keep threshold"""
    path = os.path.join(runs, "e389_rules", f"rules_card_val_s{seed}{n_sfx(n_max)}.json")
    if mode not in RULE_ROW or not os.path.exists(path):
        return None
    c = json.load(open(path))
    if any(c["data"]["route_cache"][k] != v for k, v in route_cache_identity(side["ds"].rc.manifest).items()) or c["keep_thr"] != keep_thr:
        return None
    return {k: v["rows"][RULE_ROW[mode]]["numbers"]["route_F1_road"] for k, v in c["by_delivered"].items()}


def card_path(runs, run, mode, ref_run, ref_mode, n_max):
    return os.path.join(runs, run, f"route_card_{mode}" + (f"_vs_{ref_run}" + ("" if ref_mode == mode else f"_rm{ref_mode}") if ref_run else "") + f"{n_sfx(n_max)}.json")


def score(a, runs=RUNS):
    """the card of one dump (module docstring) -> its path"""
    ref_mode = a.ref_mode or a.mode
    assert a.ref_run or not a.ref_mode, "--ref-mode is the mode of --ref-run"
    path = card_path(runs, a.run, a.mode, a.ref_run, ref_mode, a.max)
    assert not os.path.exists(path), f"a stored card is never overwritten: {path}"
    d = read_dump(runs, a.run, a.seed, a.mode, a.max)
    r = read_dump(runs, a.ref_run, a.seed, ref_mode, a.max) if a.ref_run else None
    m = d["meta"]
    side = load_side(a.seed, m["ident"], m["K"] if m.get("cache_root") else None, m.get("cache_root"), a.max)
    check_pair(d, r, side)
    pn, champ = champion_pnone(side, a.seed, a.champion, runs=RUNS)
    deliv = delivered_sets(side, pn, a.keep_thr)
    draws = draws_of(side["scene"], a.draws)
    card = dict(name=f"{a.run}_route_{a.mode}" + (f"_vs_{a.ref_run}" if a.ref_run else ""), run=a.run, seed=a.seed, mode=a.mode, split="val", frames=len(side["tokens"]), arm=m["arm"], inputs=m["inputs"], attn=m.get("attn") or "ab", dump=_file_id(d["path"]), device=m["device"], autocast=m.get("autocast"),
                keep_thr=a.keep_thr, champion=champ, data=cache_id(side), definitions=DEFINITIONS, delivered_used="kept", draws=a.draws, by_delivered={})
    if r is not None:
        card["ref"] = dict(run=a.ref_run, mode=ref_mode, arm=r["meta"]["arm"], dump=_file_id(r["path"]), device=r["meta"]["device"])
    rule = rule_reference(runs, side, a.seed, a.mode, a.keep_thr, a.max) if r is not None else None
    for name, dl in deliv.items():
        C, blk = system_numbers(side, dl, d, draws)
        if r is not None:
            Cr, rblk = system_numbers(side, dl, r, draws)
            blk.update(ref=dict(numbers=rblk["numbers"], interval=rblk["interval"]), paired=paired(C, Cr, draws),
                       gap_closed=None if rule is None else dict(rule_row=RULE_ROW[a.mode], rule_F1_road=rule[name], z_F1_road=rblk["numbers"]["route_F1_road"], m_F1_road=blk["numbers"]["route_F1_road"],
                                                                 value=gap_closed(blk["numbers"]["route_F1_road"], rblk["numbers"]["route_F1_road"], rule[name])))
        card["by_delivered"][name] = blk
        n = blk["numbers"]
        print(f"{a.run} {a.mode} [{name}] nodes {n['nodes']}  F1 road {n['route_F1_road']:.4f} lane {n['route_F1_lane']:.4f}  fork F1 {n['fork_F1_road']:.4f} ({n['fork_frames']} frames)  success30 {n['success30']:.3f}"
              + (f"  | vs {a.ref_run}/{ref_mode}: road {blk['paired']['route_F1_road']['diff']:+.4f} [{blk['paired']['route_F1_road']['lo']:+.4f}, {blk['paired']['route_F1_road']['hi']:+.4f}]" if r is not None else ""), flush=True)
    write_new(path, card)
    print(f"written {os.path.relpath(path, ROOT) if path.startswith(ROOT) else path}")
    return path


def cost(a, runs=RUNS):
    """the cost of the head for the card's clause 6: parameters and the matmul / attention FLOPs of one frame of --K tokens and --M route samples (default: the route cap of the route cache; a synthetic frame, the cost does not depend on the values), counted by torch's
    `FlopCounterMode` on the CPU; NOT timed (a time needs a quiet GPU: this is the 'derived from parameters and FLOPs' figure) -> runs/R/route_cost.json, never overwritten"""
    import jroute as JR
    import jroute_train as JRT
    import lane_route_data as LRD
    from torch.utils.flop_counter import FlopCounterMode
    path = os.path.join(runs, a.run, "route_cost.json")
    assert not os.path.exists(path), f"a stored card is never overwritten: {path}"
    m, args = JRT.load_model(a.run, "cpu", runs)
    M = a.M or json.load(open(os.path.join(LRD.route_dir("val"), "manifest.json")))["params"]["max_samples"]
    b = JRT.to_device(JR.synthetic_batch(1, a.K, M, 0), args.get("inputs") or "full", "cpu")
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(b)
    card = dict(run=a.run, arm=args.get("arm"), inputs=args.get("inputs") or "full", attn=args.get("attn") or "ab", params=sum(p.numel() for p in m.parameters()), K=a.K, M=M, gmacs=fc.get_total_flops() / 2e9, timed=False,
                note="GMACs of matmuls (matmul FLOPs / 2 of one synthetic frame on the CPU), NOT a timing; the card's 1.0 ms clause (6) stays UNVERIFIED unless a timing on a quiet GPU measured it", clause_6="UNVERIFIED (not timed)")
    write_new(path, card)
    print(f"{a.run}: {card['params']:,} parameters, {card['gmacs']:.3f} GMACs of matmuls per frame (K {a.K}, M {M}), NOT a timing: the 1.0 ms clause stays UNVERIFIED unless timed -> {os.path.relpath(path, ROOT) if path.startswith(ROOT) else path}")
    return path


def rules(a, runs=RUNS):
    """the reference rows (module docstring) on both delivered definitions -> runs/e389_rules/rules_card_val_s<S>[_n<N>].json"""
    path = os.path.join(runs, "e389_rules", f"rules_card_{a.split}_s{a.seed}{n_sfx(a.max)}.json")
    assert not os.path.exists(path), f"a stored card is never overwritten: {path}"
    modes = [x for x in a.modes.split(",") if x]
    assert modes and all(x in ("exact", "error", "gross", "wrong") for x in modes), f"--modes {a.modes}: exact, error, gross, wrong"
    side = load_side(a.seed, n_max=a.max, split=a.split, mode="wrong" if "wrong" in modes else None)
    pn, champ = champion_pnone(side, a.seed, a.champion, runs=RUNS)
    deliv = delivered_sets(side, pn, a.keep_thr)
    flags = rule_rows(side, modes, deliv)
    card = dict(name=f"rules_{a.split}_s{a.seed}", seed=a.seed, split=a.split, frames=len(side["tokens"]), keep_thr=a.keep_thr, champion=champ, data=cache_id(side), definitions=DEFINITIONS, delivered_used="kept", expect=EXPECT, tol=TOL,
                note="rows flag the road level; their lane numbers read the same flags against on_lane (no lane rule). expect = the card critic's F1_road on its proxy-delivered nodes of 788 VAL frames (the old labels); reproduced = |F1_road(proxy) - expect| <= tol",
                by_delivered={})
    for name, dl in deliv.items():
        rows = {}
        for row, fl in flags.items():
            nm = numbers(count_matrix(side, dl, fl, fl))
            rows[row] = dict(numbers=nm, **({"expect": EXPECT[row], "diff": nm["route_F1_road"] - EXPECT[row], "reproduced": abs(nm["route_F1_road"] - EXPECT[row]) <= TOL} if row in EXPECT else {}))
            print(f"rules [{name}] {row:<18} F1 road {nm['route_F1_road']:.4f} (P {nm['route_P_road']:.4f} R {nm['route_R_road']:.4f})  fork F1 {nm['fork_F1_road']:.4f}  nodes {nm['nodes']}"
                  + (f"  expect {EXPECT[row]:.3f} diff {rows[row]['diff']:+.4f}" if row in EXPECT else ""), flush=True)
        card["by_delivered"][name] = dict(rows=rows, fork=fork_info(side), delivered_tokens_per_frame=float((dl & side["valid"])[side["has_route"]].sum(1).mean()))
    write_new(path, card)
    print(f"written {os.path.relpath(path, ROOT) if path.startswith(ROOT) else path}")
    return path


# ------------------------------------------------------------------------------------------------------------------------------ selftest
def must_fail(fn, *a, **k):
    """a mutation check: fn must fail its assertion"""
    try:
        fn(*a, **k)
    except AssertionError:
        return
    raise AssertionError(f"{getattr(fn, '__name__', fn)} passed with the mutated input")


def hand_side():
    """five hand frames of ten tokens, the answers by hand in `selftest`; a token = (valid, gt_id, on_road, on_lane, x, y, delivered, p_road, p_lane). A (scene 0, fork): a padding token that has a GT id and is "delivered", an undelivered token with a GT node
    and a false token (no GT node); B (scene 0): a route node beyond 30 m, 2 of 3 near route nodes flagged; C (scene 1, fork): a success, with a flagged non-route node 35 m ahead and one 6 m behind (both outside the success window); D (scene 1): 3 of 5 near
    non-route nodes flagged, one of them at p = 0.5; E: no route"""
    T = lambda g, r, l, x, y, p, q, dl=True: (True, g, r, l, x, y, dl, p, q)  # noqa: E731
    pad = (False, -1, -1, -1, 0.0, 0.0, False, np.nan, np.nan)
    A = [T(0, 1, 1, 0, 5, .95, .85), T(1, 1, 0, 0, 10, .25, .15), T(2, 0, 0, 5, 8, .75, .65), T(3, 0, 0, 0, -10, .15, .15), T(-1, -1, -1, 0, 7, .95, .25), (False, 4, 1, 1, 0.0, 6.0, True, np.nan, np.nan), T(5, 1, 0, 0, 12, .15, .15, False), T(-1, -1, -1, 0, 9, .95, .95, False)] + [pad] * 2
    B = [T(0, 1, 1, 0, 4, .95, .95), T(1, 1, 0, 0, 14, .85, .35), T(2, 1, 0, 0, 24, .45, .25), T(3, 0, 0, 10, 10, .35, .15), T(4, 1, 0, 0, 40, .95, .95)] + [pad] * 5
    C = [T(0, 1, 1, 0, 3, .95, .95), T(1, 1, 0, 0, 8, .95, .15), T(2, 1, 0, 0, 13, .95, .15), T(3, 1, 0, 0, 18, .95, .15), T(4, 0, 0, 4, 6, .15, .15), T(5, 0, 0, -4, 12, .15, .15), T(6, 0, 0, 6, 20, .15, .15), T(7, 0, 0, 0, 35, .95, .15),
         T(8, 0, 0, 0, -6, .95, .15), pad]
    D = [T(0, 1, 1, 0, 5, .95, .95), T(1, 1, 0, 0, 10, .95, .15), T(2, 1, 0, 0, 15, .95, .15), T(3, 0, 0, 4, 5, .95, .15), T(4, 0, 0, -4, 10, .95, .15), T(5, 0, 0, 4, 15, .15, .15), T(6, 0, 0, -4, 20, .15, .15), T(7, 0, 0, 4, 25, .5, .15)] + [pad] * 2
    E = [T(0, -1, -1, 0, 5, .95, .95), T(-1, -1, -1, 0, 8, .95, .95)] + [pad] * 8
    frames = [A, B, C, D, E]
    arr = lambda j, dt: np.array([[t[j] for t in f] for f in frames], dt)  # noqa: E731
    side = dict(tokens=list("ABCDE"), scene=np.array([0, 0, 1, 1, 2]), has_route=np.array([1, 1, 1, 1, 0], bool), decision=np.array([1, 0, 1, 0, 1], bool), valid=arr(0, bool), gt_id=arr(1, int), lab_road=arr(2, np.int8), lab_lane=arr(3, np.int8),
                pos=np.stack([arr(4, float), arr(5, float)], -1), dirv=np.tile([0.0, 1.0], (5, 10, 1)))
    return side, arr(6, bool), arr(7, float), arr(8, float)


def tiled(side, deliv, pr, pl, reps=6):
    """the first four hand frames `reps` times over, one scene per copy (4 frames each): a set for the bootstrap"""
    t = lambda x: np.concatenate([x[:4]] * reps)  # noqa: E731
    big = dict(side, **{k: t(side[k]) for k in ("valid", "gt_id", "lab_road", "lab_lane", "pos", "dirv", "decision")}, has_route=np.ones(4 * reps, bool), scene=np.repeat(np.arange(reps), 4), tokens=list(range(4 * reps)))
    return big, t(deliv), t(pr), t(pl)


def selftest():
    eq = lambda x, y: abs(x - y) < 1e-9  # noqa: E731
    side, deliv, pr, pl = hand_side()
    C = count_matrix(side, deliv, flags(pr), flags(pl))
    nm = numbers(C)
    # road: A tp 1 fp 1 fn 1, B tp 3 fn 1, C tp 4 fp 2, D tp 3 fp 3 -> tp 11, fp 6, fn 2 (C: the two flagged non-route nodes beyond the window count here); lane: tp 4 (A t0, B b0, C c0, D d0), fp 2 (A t2, B b4), fn 0
    assert (nm["frames"], nm["nodes"], nm["fork_frames"]) == (4, 26, 2), nm
    assert eq(nm["route_F1_road"], 22 / 30) and eq(nm["route_P_road"], 11 / 17) and eq(nm["route_R_road"], 11 / 13), nm
    assert eq(nm["route_F1_lane"], 0.8) and eq(nm["route_P_lane"], 4 / 6) and eq(nm["route_R_lane"], 1.0), nm
    # fork frames A and C (E is a fork frame without a route: not scored): road tp 1 + 4, fp 1 + 2, fn 1
    assert eq(nm["fork_F1_road"], 10 / 14) and eq(nm["fork_P_road"], 5 / 8) and eq(nm["fork_R_road"], 5 / 6), nm
    # success: eligible B, C, D (A has two route nodes near, B's b4 is beyond 30 m), only C succeeds (B 2 of 3 flagged, D 3 of 5 non-route flagged)
    assert (nm["eligible_frames"], nm["success_frames"]) == (3, 1) and eq(nm["success30"], 1 / 3) and eq(nm["success30_all"], 1 / 4), nm
    assert nm["false_tokens"] == 1 and eq(nm["false_flag_road"], 1.0) and eq(nm["false_flag_lane"], 0.0), nm
    # a selection: frames C and D twice (a frame drawn twice counts twice): road tp 14, fp 10, fn 0
    n2 = numbers(C, np.array([2, 3, 2, 3]))
    assert n2["frames"] == 4 and eq(n2["route_F1_road"], 28 / 38) and n2["fork_frames"] == 2, n2
    # the calibration by hand (road): bin 9 holds 14 nodes of which 10 are on the route, the bins 8 7 5 4 3 2 one node each, bin 1 six nodes off the route; a covered node without a probability is refused
    cal = calibration(side, deliv, pr, side["lab_road"])
    assert [c["n"] for c in cal] == [0, 6, 1, 1, 1, 1, 0, 1, 1, 14], cal
    assert [c["share_on"] for c in cal[1:6]] == [0.0, 1.0, 0.0, 1.0, 0.0] and eq(cal[9]["share_on"], 10 / 14) and eq(cal[8]["share_on"], 1.0) and eq(cal[7]["share_on"], 0.0), cal
    must_fail(calibration, side, deliv, np.where(np.arange(10)[None] == 0, np.nan, pr), side["lab_road"])
    # the delivered definitions at their boundaries
    assert delivered_kept([1, 1, 1, 0], np.array([.59, .6, .2, .1], np.float32)).tolist() == [True, False, True, False]
    assert delivered_proxy([1, 1, 1, 0], [.5, .49, .9, .9]).tolist() == [True, False, True, False]
    # the fork clause: the hand set has 2 fork frames from 2 scenes; 70 frames from 9 scenes pass, from 7 scenes or 59 frames they are void
    assert fork_info(side)["fork_frames"] == 2 and fork_info(side)["clause"].startswith("void")
    big = dict(decision=np.ones(70, bool), has_route=np.ones(70, bool), scene=np.arange(70) % 9)
    assert fork_info(big)["clause"] == "ok" and fork_info(dict(big, scene=np.arange(70) % 7))["clause"].startswith("void") and fork_info(dict(big, decision=np.arange(70) < 59))["clause"].startswith("void")
    assert fork_info(dict(big, decision=np.arange(70) < 60, scene=np.arange(70) % 8))["clause"] == "ok", "60 fork frames from 8 scenes are the minimum"
    # the bootstrap: identical systems differ by exactly 0 on every draw (paired); a system that flags nothing is far below the other; a draw is whole scenes (6 scenes of 4 frames)
    big, dl, p_r, p_l = tiled(side, deliv, pr, pl)
    flip = np.random.RandomState(0).rand(*dl.shape) < 0.15                                   # the scenes differ from each other: a resample changes the pooled F1, so a pair on other draws would show
    Ca = count_matrix(big, dl, flags(p_r) ^ flip, flags(p_l))
    f_a = numbers(Ca)["route_F1_road"]
    assert len({round(numbers(Ca, np.flatnonzero(big["scene"] == s))["route_F1_road"], 6) for s in range(6)}) > 1, "the bootstrap scenes are identical"
    Cb = count_matrix(big, dl, np.zeros_like(dl), np.zeros_like(dl))
    dr = draws_of(big["scene"], 200)
    same = paired(Ca, Ca, dr)
    assert set(same) == set(BOOT_KEYS) and all(v["diff"] == 0 and v["lo"] == 0 and v["hi"] == 0 and v["n_draws"] == 200 for v in same.values()), same
    diff = paired(Ca, Cb, dr)
    assert diff["route_F1_road"]["lo"] > 0 and eq(diff["route_F1_road"]["diff"], f_a) and diff["route_F1_road"]["n_draws"] == 200, diff["route_F1_road"]
    assert all((np.bincount(big["scene"][s], minlength=6) % 4 == 0).all() for s in dr[:20]), "a draw is whole scenes"
    iv = interval(Ca, dr)
    assert eq(iv["route_F1_road"]["value"], f_a) and iv["route_F1_road"]["lo"] <= f_a <= iv["route_F1_road"]["hi"], iv
    # the dump writer: a padding token has NaN, a valid one its probability as float32
    fr = dump_frames(np.array([[.9, .2, .5]]), np.array([[.1, .8, .6]]), np.array([[True, False, True]]), ["tok"], [7])[0]
    assert fr["p_road"].dtype == np.float32 and np.isnan(fr["p_road"][1]) and fr["p_road"][2] == np.float32(.5) and fr["row"] == 7 and np.isnan(fr["p_lane"][1]) and fr["valid"].tolist() == [True, False, True], fr
    # the straight prior: x = 3.5 is in, 3.6 out, cos(heading, +y) = 0.7 is in, 0.6 out
    s1 = dict(pos=np.array([[[3.5, 5], [3.6, 5], [-3.5, 5], [0, 5], [0, 5]]], float), dirv=np.array([[[0, 1], [0, 1], [0.7, 0.714], [0.8, 0.6], [0.7141428, 0.7]]], float))
    assert straight_flags(s1)[0].tolist() == [True, False, True, False, True], straight_flags(s1)
    # the geometric rule on a route along +y (x = 0, y = 0 .. 20, 2 m samples), lane count 1 up to y = 8 and 3 from y = 10: half width 2.25 m and 5.75 m
    y = np.arange(0, 22, 2.0)
    route = dict(pos=np.stack([np.zeros(11), y], 1), dir=np.tile([0.0, 1.0], (11, 1)), count=np.where(y < 10, 1, 3), valid=np.ones(11, bool))
    up, down, c6, c7 = [0, 1], [0, -1], [0.8, 0.6], [0.7141428, 0.7]
    tk = np.array([[2.2, 4], [2.3, 4], [5.7, 16], [5.8, 16], [0, 30], [0, 10], [0, 10], [0, 10], [0, 10], [2.5, 8.9], [2.5, 9.1], [2.2, 5]], float)       # the last three: the nearer end's count decides (1 below y = 9, 3 above); the foot on the segment (the line is 2.2 m away, the nearest sample 2.4 m)
    hd = np.array([up, up, up, up, up, up, down, c6, c7, up, up, up], float)
    assert geometric_flags(tk, hd, route).tolist() == [True, False, True, False, False, True, False, False, True, False, True, True], geometric_flags(tk, hd, route)
    one = dict(pos=np.array([[0.0, 4.0]]), dir=np.array([up], float), count=np.array([1]), valid=np.array([True]))
    assert geometric_flags(np.array([[0, 4.0]]), hd[:1], one).tolist() == [True] and not geometric_flags(tk, hd, dict(route, valid=np.zeros(11, bool))).any()
    # a head is read only on the route cache version it was trained on (the route cap included, not the frame list: the trainer records its TRAIN cache); one without a record must not be older than the cache
    man = dict(version="v4", lane_gt="g", tokens_sha="t", params=dict(max_samples=160))
    ident = route_cache_identity(man)
    assert ident == dict(version="v4", lane_gt="g", tokens_sha="t", max_samples=160), ident
    check_trained_on(dict(route_cache=ident), 0, man, 10)
    check_trained_on(dict(route_cache=dict(ident, tokens_sha="the TRAIN frames", split="train")), 0, man, 10)
    for k, v in (("max_samples", 64), ("version", "v3"), ("lane_gt", "x")):
        must_fail(check_trained_on, dict(route_cache=dict(ident, **{k: v})), 20, man, 10)
    check_trained_on({}, 10, man, 10)
    must_fail(check_trained_on, {}, 9.9, man, 10)
    # a dump and its reference dump are made on the route cache that is read: each of the four identity keys, for the dump and for the reference
    for k, v in (("max_samples", 64), ("version", "v3"), ("lane_gt", "x"), ("tokens_sha", "u")):
        must_fail(check_route_cache, dict(ident, **{k: v}), ident, "the dump")
    check_route_cache(ident, ident, "the dump")
    from types import SimpleNamespace
    st = dict(tokens=["a"], valid=np.ones((1, 3), bool), ds=SimpleNamespace(rc=SimpleNamespace(manifest=man, dir=os.path.join(ROOT, "runs", "r")), tok=SimpleNamespace(dir=os.path.join(ROOT, "runs", "t"))))
    dump_of = lambda rc: dict(tokens=["a"], valid=np.ones((1, 3), bool), meta=dict(seed=0, cache="c", route_cache=rc))  # noqa: E731
    check_pair(dump_of(ident), dump_of(ident), st)
    check_pair(dump_of(ident), None, st)
    must_fail(check_pair, dump_of(ident), dump_of(dict(ident, max_samples=64)), st)
    must_fail(check_pair, dump_of(dict(ident, version="v3")), dump_of(ident), st)
    must_fail(check_pair, dump_of(ident), dump_of(dict(ident, tokens_sha="u")), st)
    assert eq(gap_closed(0.8, 0.5, 0.9), 0.75) and np.isnan(gap_closed(0.7, 0.9, 0.9)), "the share of the gap between z and the rule that m closes"
    # a card is never overwritten
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        p = write_new(os.path.join(t, "c.json"), dict(x=np.float32(1.5), y=np.arange(2)))
        assert json.load(open(p)) == dict(x=1.5, y=[0, 1])
        must_fail(write_new, p, {})
    print("jroute_eval selftest ok")


# ------------------------------------------------------------------------------------------------------------------------------ command line
def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("dump")
    s.add_argument("--run", required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--mode", required=True, choices=MODES)
    s.add_argument("--cpu", action="store_true", help="the head is tiny: the CPU is enough (default: the GPU when one is visible)")
    s.add_argument("--bs", type=int, default=8)
    s.add_argument("--workers", type=int, default=0)
    s.add_argument("--max", type=int, default=0, help="only the first N frames (a test; the names carry _n<N>)")
    s = sub.add_parser("score")
    s.add_argument("--run", required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--mode", required=True, choices=MODES)
    s.add_argument("--ref-run", default=None, help="the matched reference (arm z): its dump of --ref-mode; the paired intervals are this run minus it")
    s.add_argument("--ref-mode", default=None, choices=MODES, help="the mode of --ref-run's dump (default: --mode); the name gets _rm<mode> when it differs")
    s.add_argument("--champion", default=None, help="the run of the delivered set (default e373_cb_s<seed>); its token_val.pkl")
    s.add_argument("--keep-thr", type=float, default=KEEP_THR, help="a token is delivered at p_none < K (the D115 decode: 0.6)")
    s.add_argument("--draws", type=int, default=DRAWS)
    s.add_argument("--max", type=int, default=0)
    s = sub.add_parser("rules")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--split", default="val", choices=("val",))
    s.add_argument("--modes", default="exact,error,gross,wrong", help="the route tokens of the geometric rule: exact, error, gross, wrong (comma separated); wrong = the route of another scene, the rule's behaviour on a wrong map")
    s.add_argument("--champion", default=None)
    s.add_argument("--keep-thr", type=float, default=KEEP_THR)
    s.add_argument("--max", type=int, default=0)
    s = sub.add_parser("cost")
    s.add_argument("--run", required=True)
    s.add_argument("--K", type=int, default=512, help="tokens per frame (the cache's K)")
    s.add_argument("--M", type=int, default=0, help="route samples per frame (default: the route cap of the route cache)")
    sub.add_parser("selftest")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    torch.set_num_threads(2)
    {"dump": dump, "score": score, "rules": rules, "cost": cost, "selftest": lambda _: selftest()}[a.cmd](a)


if __name__ == "__main__":
    main()
