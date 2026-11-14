"""soa_maps - our map layers against the published camera + radar map model GaussianCaR on the SAME frames, the SAME ground truth, the SAME camera cells and the SAME metric code
(evaluation/metrics.SegMeter), with the paired scene bootstrap of the difference per row (card t26, G0; promoted from the scratch temp/t16/soa_maps_v8.py, used a second time).

    python evaluation/soa_maps.py RUN OUT.json [--n N] [--frames-from STORED.json] [--device cpu|cuda] [--no-model] [--gt-signature SIG] [--allow-gt-mismatch]
                                  [--offset 0.5] [--x-max 50] [--boot 1000] [--seed 0] [--check STORED.json]
    python evaluation/soa_maps.py --retable OUT_frames.npz OUT.json     the rows again from the stored per-frame counts (no model, no data)
    python evaluation/soa_maps.py --selftest | --mutants                synthetic known answers | the planted bugs the self-test must catch (CPU, no data)

What it scores. RUN = a run folder (its last checkpoint under the EMA weights, the run's own code snapshot, the suite's cold state: an EMPTY memory before every frame, dt 0), run through
`bev_eval.free_source` / `FreeSubject` (the network never reads the ground truth). GaussianCaR: the logits of the lane project's cache (`~/exp_gcar_b123/feats/val_logit7.npy`, 1706 frames of
43 of our 82 val scenes; the ones that are samples of the run's val set are used), drivable_area = max(sigmoid(lane), sigmoid(road_segment)), the other five layers directly; array row = -y,
column = forward + OFFSET (0.5 m, temp/soa_gap/align_gcar.py), 200 x 200 cells of 0.5 m, +-50 m. Six layers, tolerances 0 / 0.5 / 1 m, thresholds 0.3 / 0.5 / 0.7; counts per frame (additive).
GT: the stored GT of a signature (`--gt-signature`, default the live generator's; GtGenerator.store, nothing is generated or written); printed with its version, and REFUSED when it is not the
GT the run was trained on (runs/RUN/gt.json) unless `--allow-gt-mismatch` (a base scored again on a newer GT says so in the json).

Two regions, both over the GT camera cells (`fov`):
  claim  the cells that have a reference cell: forward below the end of the reference array (49.5 m of our grid for OFFSET 0.5, a cell row at 49.5 - 50 m has none) and inside it laterally.
         All maps are CROPPED to these rows before the counting, so that nothing beyond them reaches the tolerance dilation. A request for more than 50 m is refused.
  old    all 0-60 m, the reference indices clipped into the array (the cells beyond 49.5 m read its last column): "old, for reproduction of table 1 only", never a claim.
Per row (layer, tolerance): n_gt cells of the region; ours and the reference at threshold 0.5 and their ratio; d1 = ours - reference with both at threshold 0.5; d2 = each system at its OWN best
threshold of {0.3, 0.5, 0.7} (chosen once on all frames: in-sample, labelled so; the interval is conditional on that choice). Each difference has the paired scene bootstrap interval
(`Evaluation.stats.bootstrap_scenes`: 1000 draws, seed 0, scenes drawn with replacement, the same draw for both systems; the helper of `bev_suite.py compare`). Output: OUT.json (per row: n_gt,
ours, reference, d1 / d2 with interval), OUT.md, OUT_frames.npz (the per-frame counts, so a row can be bootstrapped again: --retable).

Reproduction gate (GT v8, runs/t16_full_s0, the 384 frames of temp/t16/soa/soa_maps_v8_t16_full_s0.json): `--frames-from` that json, `--gt-signature 6a7aa7864a`, `--check` it: every per-frame count of the
region `old` equals the stored one (CPU against CPU: exactly; a GPU run differs in the cells that sit on a threshold), and the reference column of Table 1 is reproduced to 0.001 on all frames
with `--no-model`. 2.5 s per frame on 2 CPU threads, so `--n 40` for the model.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
WS = os.path.abspath(os.path.join(ROOT, ".."))
for _p in (WS, os.path.join(WS, "autoresearch"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from Evaluation.comparison import maps as comparison_maps  # noqa: E402
from Evaluation.stats import bootstrap_scenes, scene_picks  # noqa: E402
from evaluation.metrics import TOLERANCES, SegMeter  # noqa: E402

LAYERS = ("drivable_area", "ped_crossing", "walkway", "carpark_area", "lane_divider", "road_divider")
THR = (0.3, 0.5, 0.7)
SYSTEMS = ("ours", "ref")                                   # ref = GaussianCaR
CARD_ROWS = (("drivable_area", 0.0), ("walkway", 0.0), ("carpark_area", 0.0), ("ped_crossing", 0.0), ("lane_divider", 0.5), ("road_divider", 0.5))      # the rows of the card (section 1)
GC = {"lane": 0, "road_segment": 1, "road_divider": 2, "lane_divider": 3, "ped_crossing": 4, "walkway": 5, "carpark_area": 6}      # the channels of the cached logits [7, 200, 200]
NX, NY, RES, X_MIN, Y_MIN = 120, 104, 0.5, 0.0, -26.0       # the BEV grid of the front GT
GRID = SimpleNamespace(x_min=X_MIN, res=RES)
REF_N, REF_HALF = 200, 50.0                                 # the reference array: 200 x 200 cells, +-50 m
OFFSET = 0.5                                                # forward offset of the reference array against our grid (m)
CLAIM_X_MAX = 50.0
REGIONS = {"claim": "claim region: camera cells that have a reference cell (forward below the reference array's end, inside it laterally), maps cropped to them",
           "old": "old, for reproduction of table 1 only: camera cells 0-60 m, reference indices clipped into its array (cells beyond 49.5 m read its last column)"}
HOME = os.path.expanduser("~")
LOGITS = f"{HOME}/exp_gcar_b123/feats/val_logit7.npy"
TOKENS = f"{HOME}/exp_gcar_b123/feats/val_tokens.json"


# ------------------------------------------------------------------------------------------------ the reference's maps on our grid
def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(a, np.float32), -30, 30)))


def reference_index(offset=None):
    """(row, column) of the reference array for every cell of our grid, NOT clipped: row = -y (left), column = forward + offset; 0.5 m cells, +-50 m"""
    offset = OFFSET if offset is None else offset
    xs = X_MIN + (np.arange(NX) + 0.5) * RES
    ys = Y_MIN + (np.arange(NY) + 0.5) * RES
    X, Y = np.meshgrid(xs + offset, ys, indexing="ij")
    return np.floor((-Y + REF_HALF) / RES).astype(int), np.floor((X + REF_HALF) / RES).astype(int)


def inside_array(row, col):
    return (row >= 0) & (row < REF_N) & (col >= 0) & (col < REF_N)


def claim_rows(offset=None, x_max=CLAIM_X_MAX):
    """how many leading rows of our grid (x forward) have a reference cell for every cell and lie below x_max. Refuses what would score cells that the reference does not cover: x_max beyond the
    array's 50 m, or a row below x_max that is only partly inside the array laterally"""
    if x_max > REF_HALF:
        raise ValueError(f"x_max {x_max:g} m is beyond the reference array (+-{REF_HALF:g} m): its cells there do not exist")
    row, col = reference_index(offset)
    ok = inside_array(row, col)
    xs = X_MIN + (np.arange(NX) + 0.5) * RES
    below = xs < x_max
    partly = below & ok.any(1) & ~ok.all(1)
    if partly.any():
        raise ValueError(f"{int(partly.sum())} rows of the grid below x {x_max:g} m are only partly inside the reference array (lateral): refused")
    n = int((below & ok.all(1)).sum())
    if n == 0 or not (below & ok.all(1))[:n].all():
        raise ValueError("the claim region is empty or not a block of leading rows")
    return n


def reference_maps(logits, offset=None):
    """cached logits [7, 200, 200] -> probabilities of our six layers on our grid (indices clipped into the array: only the region `old` scores the cells that had to be clipped)"""
    row, col = reference_index(offset)
    r, c = np.clip(row, 0, REF_N - 1), np.clip(col, 0, REF_N - 1)
    p = {k: sigmoid(logits[i])[r, c] for k, i in GC.items()}
    p["drivable_area"] = np.maximum(p["lane"], p["road_segment"])
    return {k: p[k] for k in LAYERS}


# ------------------------------------------------------------------------------------------------ counts per frame
def frame_counts(maps, gt, fov, rows):
    """one frame. maps = {system: {layer: probability (X, Y)}}, gt = {layer: coverage (X, Y)}, fov = camera cells (X, Y) bool, rows = the claim region's leading rows.
    -> ({(region, system): counts (thresholds, layers, tolerances, 4) = [p_hit, p_miss, g_hit, g_miss]}, {region: n_gt (layers,)}); the counting is SegMeter's: a map predicted at threshold th is
    binary, the tolerance dilation of the cropped maps, the cells of fov"""
    cnt, n_gt = {}, {}
    for region, nr in (("claim", rows), ("old", NX)):
        g, f = {k: np.asarray(gt[k])[:nr] for k in LAYERS}, np.asarray(fov)[:nr]
        for system, probs in maps.items():
            arr = np.zeros((len(THR), len(LAYERS), len(TOLERANCES), 4))
            for ti, th in enumerate(THR):
                meter = SegMeter(GRID)
                meter.add({k: (np.asarray(probs[k])[:nr] > th).astype(np.float32) for k in LAYERS}, g, mask=f)
                for li, k in enumerate(LAYERS):
                    arr[ti, li] = meter.c[(0.0, k, "all", "all")]["cnt"]
            cnt[(region, system)] = arr
        meter = SegMeter(GRID)
        meter.add({k: np.zeros_like(g[k]) for k in LAYERS}, g, mask=f)
        n_gt[region] = np.array([meter.c[(0.0, k, "all", "all")]["n_gt"] for k in LAYERS], float)
    return cnt, n_gt


class Scores:
    """the per-frame additive counts of one scoring: cnt[(region, system)] (frames, thresholds, layers, tolerances, 4), n_gt[region] (frames, layers), tokens, scenes"""

    def __init__(self, tokens, scenes, cnt, n_gt):
        self.tokens, self.scenes, self.cnt, self.n_gt = list(tokens), list(scenes), cnt, n_gt

    @classmethod
    def stack(cls, tokens, scenes, frames):
        """frames = the list of frame_counts results"""
        cnt = {k: np.stack([f[0][k] for f in frames]) for k in frames[0][0]}
        return cls(tokens, scenes, cnt, {r: np.stack([f[1][r] for f in frames]) for r in frames[0][1]})

    def save(self, path):
        np.savez_compressed(path, tokens=np.array(self.tokens), scenes=np.array(self.scenes),
                            **{f"cnt|{r}|{s}": v for (r, s), v in self.cnt.items()}, **{f"n_gt|{r}": v for r, v in self.n_gt.items()})

    @classmethod
    def load(cls, path):
        with np.load(path) as z:
            cnt = {tuple(k.split("|")[1:]): z[k] for k in z.files if k.startswith("cnt|")}
            return cls(z["tokens"].tolist(), z["scenes"].tolist(), cnt, {k.split("|")[1]: z[k] for k in z.files if k.startswith("n_gt|")})

    def systems(self, region):
        return [s for s in SYSTEMS if (region, s) in self.cnt]


# ------------------------------------------------------------------------------------------------ rows and intervals
def iou(counts):
    """the tolerant IoU of additive counts [p_hit, p_miss, g_hit, g_miss] (the metric's own formula: Evaluation.comparison.maps.tolerant_scores)"""
    return comparison_maps.tolerant_scores(counts)["iou"]


def best_thresholds(total):
    """per system, the index into THR of the best threshold of every (layer, tolerance) on the summed counts total (thresholds, layers, tolerances, 4); ties go to the higher threshold,
    a row without ground truth to 0.5 (the stored table 2's rule)"""
    best = np.full(total.shape[1:3], THR.index(0.5))
    for li in range(total.shape[1]):
        for ti in range(total.shape[2]):
            v = [(iou(total[k, li, ti]), THR[k]) for k in range(len(THR))]
            v = [x for x in v if x[0] == x[0]]
            if v:
                best[li, ti] = THR.index(max(v)[1])
    return best


def by_scene(vectors, scenes):
    """{scene: [per-frame vector]}: the unit of the bootstrap is the SCENE (the frames of a scene are not independent)"""
    out = {}
    for v, s in zip(vectors, scenes):
        out.setdefault(s, []).append(v)
    return out


def bootstrap(ref_by, ours_by, reduce, keys, n, seed):
    """the paired scene bootstrap of ours - reference (the helper of `bev_suite.py compare`: a = control = the reference, b = ours; diff = b - a)"""
    return bootstrap_scenes(ref_by, ours_by, reduce, keys, n=n, seed=seed)


def rows_of(scores, region, boot=1000, seed=0):
    """the rows of one region: [{layer, tol, card, n_gt, ours, ref, ratio, d1, d2}]: ours / ref at threshold 0.5; d1 = ours - ref at 0.5; d2 = each system at its own best threshold, each with the paired scene
    bootstrap interval {diff, lo, hi, n_valid, ours, ref}; best = {system: {iou, thr}} each system's best threshold (the d2 numbers); without `ours` (a reference-only run) only the reference's numbers"""
    systems = scores.systems(region)
    total = {s: scores.cnt[(region, s)].sum(0) for s in systems}
    best = {s: best_thresholds(total[s]) for s in systems}
    t05 = THR.index(0.5)
    n_gt = scores.n_gt[region].sum(0)
    cols = [(d, li, ti) for d in (0, 1) for li in range(len(LAYERS)) for ti in range(len(TOLERANCES))]        # the keys: d1 (threshold 0.5) then d2 (each system's best), per layer and tolerance
    keys = [f"d{d + 1}|{li}|{ti}" for d, li, ti in cols]
    where = {k: i for i, k in enumerate(keys)}

    def vectors(s):                                                   # (frames, keys, 4): the counts each key reads from a frame
        c = scores.cnt[(region, s)]
        return np.stack([c[:, t05 if d == 0 else best[s][li, ti], li, ti] for d, li, ti in cols], 1)

    def reduce(counts, keys=keys):
        tot = np.sum(counts, axis=0) if len(counts) else np.zeros((len(where), 4))
        return {k: iou(tot[where[k]]) for k in keys}
    res = {}
    if "ours" in systems and "ref" in systems:
        res = bootstrap(by_scene(vectors("ref"), scores.scenes), by_scene(vectors("ours"), scores.scenes), reduce, keys, boot, seed)
    out = []
    for li, layer in enumerate(LAYERS):
        for ti, tol in enumerate(TOLERANCES):
            row = {"layer": layer, "tol": tol, "card": (layer, tol) in CARD_ROWS, "n_gt": int(n_gt[li])}
            for s in systems:
                row[s] = iou(total[s][t05, li, ti])
            row["best"] = {s: {"iou": iou(total[s][best[s][li, ti], li, ti]), "thr": THR[best[s][li, ti]]} for s in systems}
            if "ours" in row and "ref" in row:
                row["ratio"] = row["ours"] / row["ref"] if row["ref"] == row["ref"] and row["ref"] else float("nan")
                for d in ("d1", "d2"):
                    r = res.get(f"{d}|{li}|{ti}")
                    if r is not None:
                        row[d] = {"ours": r["b"], "ref": r["a"], "diff": r["diff"], "lo": r["lo"], "hi": r["hi"], "n_valid": r["n_valid"], **({"in_sample_thresholds": True} if d == "d2" else {})}
            out.append(row)
    return out


def fmt(v, d=3):
    return "n/a" if v is None or v != v else f"{v:.{d}f}"


def interval(d):
    return "n/a" if not d else f"{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"


def markdown(meta, tables):
    L = [f"# Map layers against GaussianCaR: {meta['run'] + ' (ours)' if meta.get('run') and meta.get('device') != 'none' else 'the reference only'}, GT {meta.get('gt', {}).get('version')} ({meta.get('gt', {}).get('signature')}), VAL", "",
         f"{meta['n_frames']} frames of {meta['n_scenes']} scenes ({meta['frames_rule']}); camera cells; cold (empty memory before every frame), dt 0; reference forward offset {meta['offset_m']} m. "
         "Row = layer and tolerance (IoU@t: a cell is right if the other map has a cell within t m). n_gt = GT cells of the region. d1 = ours - reference, both at threshold 0.5; d2 = each at its OWN best of "
         "{0.3, 0.5, 0.7}, chosen once on these frames (IN-SAMPLE; the interval is conditional on the choice). Interval: paired scene bootstrap (%d draws, seed %d), 95 %%. VAL, not TEST." % (meta["boot"], meta["seed"])]
    for region, rows in tables.items():
        L += ["", f"## {REGIONS[region]}", ""]
        for title, pick in (("card rows", lambda r: r["card"]), ("all rows", lambda r: True)):
            L += [f"### {title}", "", "| layer | tol m | n_gt cells | ours | reference | ours / ref | d1 = ours - ref [95 %] | ours best | ref best | d2 [95 %] |", "|---|---|---|---|---|---|---|---|---|---|"]
            for r in (x for x in rows if pick(x)):
                b = r.get("best", {})
                L.append(f"| {r['layer']} | {r['tol']:g} | {r['n_gt']} | {fmt(r.get('ours'))} | {fmt(r.get('ref'))} | {fmt(r.get('ratio'), 2)} | {interval(r.get('d1'))} | "
                         f"{fmt(b.get('ours', {}).get('iou'))} (thr {b.get('ours', {}).get('thr', '-')}) | {fmt(b.get('ref', {}).get('iou'))} (thr {b.get('ref', {}).get('thr', '-')}) | {interval(r.get('d2'))} |")
            L.append("")
    L.append(f"Command: `{meta.get('command')}`; {meta.get('seconds_per_frame', 0):.1f} s per frame on {meta.get('device')}; GT: {meta.get('gt', {}).get('note', '')}")
    return "\n".join(L) + "\n"


def write_outputs(out_json, meta, scores, boot, seed):
    tables = {r: rows_of(scores, r, boot, seed) for r in ("claim", "old")}
    stem = out_json[:-5] if out_json.endswith(".json") else out_json
    with open(out_json, "w") as f:
        json.dump({"meta": meta, "regions": {r: {"label": REGIONS[r], "rows": t} for r, t in tables.items()}}, f, indent=1, default=float)
    scores.save(stem + "_frames.npz")
    with open(stem + ".md", "w") as f:
        f.write(markdown(meta, tables))
    return tables


# ------------------------------------------------------------------------------------------------ the data: frames, GT, our network
def trained_gt(run):
    """the GT signature a run was trained on (RUN/gt.json), None when that cannot be told: no run, no folder, no file, no key"""
    try:
        with open(os.path.join(run, "gt.json")) as f:
            sig = json.load(f).get("gt_signature")
    except (TypeError, OSError, ValueError, AttributeError):
        return None
    return sig if isinstance(sig, str) and sig else None


def check_gt(sig, trained, allow_mismatch=False):
    """True = the GT scored against (signature `sig`) is the GT the run was trained on (`trained`: its cache key or a signature; a version that makes the same GT counts), False = another one,
    None = unknown. Another GT AND an unknown one are REFUSED (SystemExit) unless allow_mismatch: an unchecked identity is no identity"""
    from GtGenerator import versions
    match = None if trained is None else versions.canonical(trained.rsplit("_", 1)[-1]) == versions.canonical(sig)
    if match is not True and not allow_mismatch:
        what = "the GT the run was trained on is unknown (no runs/RUN/gt.json with a gt_signature)" if match is None else f"the run was trained on {trained}"
        raise SystemExit(f"refused: {what}, the GT scored against is {sig}; pass --allow-gt-mismatch to score the run on it anyway (the json records it)")
    return match


def gt_identity(run, signature=None, allow_mismatch=False):
    """the GT that is scored against: the stored GT of `signature` (default the live generator's), printed with its version; refused (before anything is read) when it is not the GT the run was
    trained on (runs/RUN/gt.json) or when that is unknown, unless allow_mismatch; -> {signature, version, store, run_signature, match, note}"""
    from GtGenerator import current_signature, versions
    sig = signature or current_signature()
    trained = trained_gt(run)
    match = check_gt(sig, trained, allow_mismatch)
    store = os.path.join(WS, "data", "autogt", versions.canonical(sig))
    if not os.path.isdir(store):
        raise SystemExit(f"no stored GT {store} for the signature {sig}")
    info = {"signature": sig, "version": versions.version_of_key(sig), "store": os.path.relpath(store, WS), "run_signature": trained, "match": match,
            "note": "the live GT" if signature is None else f"the GT of signature {sig}"}
    print(f"GT scored against: {info['version']} signature {sig} (store {info['store']}); the run was trained on {trained} -> " + {True: "the same GT", None: "UNKNOWN", False: "A DIFFERENT GT"}[match], flush=True)
    if match is not True:
        info["note"] += f"; the run was trained on {trained or 'an unknown GT'} (--allow-gt-mismatch)"
    return info


def select_frames(universe, n=None, only=None):
    """universe = [{i (row of the logits), token, scene}]; only = the tokens of a stored result (their order); n = that many frames spread evenly over what is left.
    -> (the frames, the tokens of `only` that are not in the universe: they cannot be scored and are reported, never dropped silently)"""
    have = {u["token"] for u in universe}
    missing = [t for t in (only or []) if t not in have]
    sel = [u for u in universe if only is None or u["token"] in only]
    if only is not None:
        order = {t: k for k, t in enumerate(only)}
        sel.sort(key=lambda u: order[u["token"]])
    if n and n < len(sel):
        sel = sel[:: max(1, len(sel) // n)][:n]
    return sel, missing


@contextlib.contextmanager
def light_devkit(scene_names):
    """the training code builds its datasets on the nuScenes devkit: here the memory-light one of GtGenerator (CAM_FRONT + radars of the given scenes, 0.7 GB instead of 8 GB; the box is shared)"""
    import nuscenes.nuscenes as NN
    from GtGenerator.light_nusc import light_nuscenes
    real = NN.NuScenes

    def light(version, dataroot=None, verbose=False, **kw):
        NN.NuScenes = real                                          # light_nuscenes subclasses the real class
        try:
            return light_nuscenes(dataroot, version, scenes=scene_names)
        finally:
            NN.NuScenes = light
    NN.NuScenes = light
    try:
        yield
    finally:
        NN.NuScenes = real


def our_network(run, device):
    """(frames: {token: (scene, FrameRef)}, predict(token) -> {layer: probability map} at dt 0 from an empty memory, describe: the Subject's record, source): the run's last checkpoint under its EMA weights, no GT read"""
    os.chdir(ROOT)
    from evaluation import bev_eval
    from evaluation.bev_free import FreeSubject
    cfg = json.load(open(os.path.join(run, "config.json")))
    names = cfg.get("val_scenes") or json.load(open(os.path.join(ROOT, "data", "scenes_parts1_6.json")))["val"]
    with light_devkit(names):
        src = bev_eval.free_source(run, "val", workers=0)
    subj = FreeSubject(bev_eval.subject(run, src, device), dts=(0.0,))
    refs = {r.key: r for sc in src.scenes() for r in src.frames(sc)}

    def predict(token):
        subj.reset()                                                # the suite's state "reset": an empty memory before every frame
        out = subj.predict(src.load(refs[token]))
        return {k: np.asarray(v) for k, v in out.outputs["dt"][0.0]["prob"].items() if k in LAYERS}
    return refs, predict, subj.describe(), src


def collect(a):
    """the scoring of the frames chosen by the arguments -> (Scores, meta)"""
    t0 = time.time()
    run = os.path.relpath(os.path.abspath(a.run), ROOT).replace("\\", "/") if a.run else None
    os.chdir(ROOT)
    from GtGenerator.store import load_keyframe
    gt = gt_identity(run, a.gt_signature, a.allow_gt_mismatch)
    rows = claim_rows(a.offset, a.x_max)
    tokens = json.load(open(a.tokens))
    stored = json.load(open(a.frames_from)) if a.frames_from else None
    only = [f["token"] for f in stored["frames"]] if stored else None
    if a.no_model:
        if stored is None:
            raise SystemExit("--no-model needs --frames-from (the frames are the run's val samples: the first and last keyframe of a scene are not)")
        universe = [{"i": f["i"], "token": f["token"], "scene": f["scene"]} for f in stored["frames"]]
        predict = describe = None
    else:
        refs, predict, describe, src = our_network(run, a.device)
        have = {t: i for i, t in enumerate(tokens)}
        universe = sorted(({"i": have[t], "token": t, "scene": r.scene} for t, r in refs.items() if t in have), key=lambda u: u["i"])
        print(f"{len(tokens)} cached reference frames, {len(universe)} are samples of the run's val set", flush=True)
    sel, missing = select_frames(universe, a.n, only)
    if missing:
        print(f"WARNING: {len(missing)} of the {len(only)} frames of {a.frames_from} are not samples of the run's val set (or have no cached reference logits) and are NOT scored", flush=True)
    if not sel:
        raise SystemExit("no frame to score")
    logits = np.load(a.logits, mmap_mode="r")
    store = os.path.join(WS, gt["store"])
    frames = []
    for k, u in enumerate(sel):
        g = load_keyframe(store, u["token"])
        maps = {"ref": reference_maps(np.asarray(logits[u["i"]]), a.offset)}
        if predict:
            maps["ours"] = predict(u["token"])
        frames.append(frame_counts(maps, g, np.asarray(g["camera"], np.float32) > 0, rows))
        if (k + 1) % 5 == 0 or k + 1 == len(sel):
            el = time.time() - t0
            print(f"{k + 1}/{len(sel)} frames, {el / (k + 1):.1f} s per frame", flush=True)
    scores = Scores.stack([u["token"] for u in sel], [u["scene"] for u in sel], frames)
    meta = {"script": "evaluation/soa_maps.py", "command": " ".join(sys.argv), "date": datetime.datetime.now().isoformat(timespec="seconds"), "run": run, "subject": describe,
            "gt": gt, "n_frames": len(sel), "n_scenes": len(set(scores.scenes)), "offset_m": a.offset, "claim_rows": rows, "frames_requested": len(only) if only else None, "frames_missing": len(missing), "frames_missing_tokens": missing[:10], "x_max": a.x_max, "thresholds": list(THR), "boot": a.boot, "seed": a.seed,
            "frames_rule": (f"the frames of {a.frames_from}" if a.frames_from else "every frame of the run's val set that has cached reference logits") + (f", {len(sel)} spread evenly" if a.n else ""),
            "device": "none" if a.no_model else a.device, "seconds_per_frame": (time.time() - t0) / len(sel), "logits": a.logits, "tokens": a.tokens}
    return scores, meta


def check_stored(scores, stored_json):
    """the reproduction gate: the per-frame counts of region `old` against those of the stored scratch run (temp/t16/soa_maps_v8.py: `_frames.npz` next to its json), frame by frame (by token),
    for every system both have; and, when the frames are the same, the reference's numbers of its Table 1. -> {system: max abs count difference} and the table difference"""
    stem = stored_json[:-5]
    meta = json.load(open(stored_json))
    names = {"ours": "ours", "ref": "gcar"}
    with np.load(stem + "_frames.npz") as z:
        pos = {f["token"]: k for k, f in enumerate(meta["frames"])}
        idx = [pos[t] for t in scores.tokens if t in pos]
        mine = [k for k, t in enumerate(scores.tokens) if t in pos]
        out = {"frames_compared": len(idx)}
        for s in scores.systems("old"):
            worst = 0.0
            for ti, th in enumerate(THR):
                for li, layer in enumerate(LAYERS):
                    st = z[f"{names[s]}|{th}|+0.00|{layer}|all|all|cnt"][idx]
                    worst = max(worst, float(np.abs(st - scores.cnt[("old", s)][mine, ti, li]).max()))
            out[s] = worst
        sys_json = meta["systems"]
        if len(idx) == len(meta["frames"]) == len(scores.tokens):
            diff = diff2 = 0.0
            for s in scores.systems("old"):
                for li, layer in enumerate(LAYERS):
                    for ti, tol in enumerate(("iou@0", "iou@0.5", "iou@1")):
                        want = sys_json[names[s]][layer]["all"]["0.5"][tol]
                        got = iou(scores.cnt[("old", s)].sum(0)[THR.index(0.5), li, ti])
                        diff = max(diff, abs(want - got))
                        best = max(v for v in (sys_json[names[s]][layer]["all"][str(th)][tol] for th in THR) if v is not None)
                        diff2 = max(diff2, abs(best - max(iou(scores.cnt[("old", s)].sum(0)[k, li, ti]) for k in range(len(THR)))))
            out["table1_max_diff"], out["table2_max_diff"] = diff, diff2
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], allow_abbrev=False)
    p.add_argument("run", nargs="?")
    p.add_argument("out", nargs="?")
    p.add_argument("--n", type=int, default=0, help="that many frames, spread evenly (0 = all)")
    p.add_argument("--frames-from", default=None, help="a stored result json: score exactly its frames")
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-model", action="store_true", help="the reference only (needs --frames-from)")
    p.add_argument("--gt-signature", default=None, help="the stored GT of this signature (default: the live generator's)")
    p.add_argument("--allow-gt-mismatch", action="store_true")
    p.add_argument("--offset", type=float, default=OFFSET)
    p.add_argument("--x-max", type=float, default=CLAIM_X_MAX)
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logits", default=LOGITS)
    p.add_argument("--tokens", default=TOKENS)
    p.add_argument("--check", default=None, help="a stored result json (temp/t16/soa/soa_maps_v8_*.json): the reproduction gate")
    p.add_argument("--retable", default=None, help="OUT_frames.npz: the rows again from stored per-frame counts (the second argument is the json to write)")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--mutants", action="store_true")
    a = p.parse_args(argv)
    if a.selftest:
        print(selftest())
        return 0
    if a.mutants:
        return mutants()
    if a.retable:
        scores = Scores.load(a.retable)
        out = a.run if a.run else a.retable.replace("_frames.npz", "_retable.json")
        write_outputs(out, {"run": None, "n_frames": len(scores.tokens), "n_scenes": len(set(scores.scenes)), "frames_rule": f"the stored counts {a.retable}", "offset_m": a.offset,
                            "boot": a.boot, "seed": a.seed, "command": " ".join(sys.argv), "device": "none"}, scores, a.boot, a.seed)
        print(f"wrote {out}")
        return 0
    if not a.out or (not a.run and not a.no_model):
        p.error("RUN and OUT.json are required")
    for k in ("out", "frames_from", "check", "logits", "tokens"):                   # absolute, the run's folder is entered
        if getattr(a, k):
            setattr(a, k, os.path.abspath(getattr(a, k)))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    scores, meta = collect(a)
    tables = write_outputs(a.out, meta, scores, a.boot, a.seed)
    print(f"wrote {a.out} (+ .md, _frames.npz): {meta['n_frames']} frames, {meta['n_scenes']} scenes")
    for r in tables["claim"]:
        if r["card"]:
            print(f"claim {r['layer']:14s} iou@{r['tol']:g}  n_gt {r['n_gt']:8d}  ours {fmt(r.get('ours'))}  ref {fmt(r.get('ref'))}  d1 {interval(r.get('d1'))}  d2 {interval(r.get('d2'))}")
    if a.check:
        print("check against", a.check, check_stored(scores, a.check))
    return 0


# ------------------------------------------------------------------------------------------------ self-test and the planted bugs
def _plant_world(seed=0, n_scenes=4, per_scene=3):
    """synthetic frames with known answers. Our grid: a road band, walkway beside it, a crossing, a car park, a lane divider line and two road dividers, a camera wedge. The reference's logits are
    written by the HAND-DERIVED mapping (row = 151 - iy, column = ix + 101 for the offset 0.5), so a wrong orientation / offset / channel makes the table wrong. Ours: GT with planted errors per scene."""
    rng = np.random.default_rng(seed)
    X, Y = np.meshgrid(X_MIN + (np.arange(NX) + 0.5) * RES, Y_MIN + (np.arange(NY) + 0.5) * RES, indexing="ij")
    frames = []
    for sc in range(n_scenes):
        for k in range(per_scene):
            yc = 2.0 * np.sin(X / 12.0 + rng.uniform(0, 6))
            gt = {"drivable_area": (np.abs(Y - yc) < 5.0).astype(np.float32), "walkway": ((np.abs(Y - yc) >= 5.0) & (np.abs(Y - yc) < 8.0)).astype(np.float32),
                  "ped_crossing": ((X > 20) & (X < 23) & (np.abs(Y - yc) < 5)).astype(np.float32), "carpark_area": ((X > 30) & (X < 36) & (Y > 9) & (Y < 14)).astype(np.float32),
                  "lane_divider": (np.abs(Y - yc) < 0.25).astype(np.float32), "road_divider": (np.abs(np.abs(Y - yc) - 5.0) < 0.25).astype(np.float32)}
            fov = (np.abs(np.arctan2(Y, X)) <= 0.6) & (X > 0)
            frames.append((sc, gt, fov))
    return frames


def _reference_logits(gt):
    """the logits [7, 200, 200] of a reference that is right on GT, written with the hand-derived inverse mapping (row = 151 - iy, column = ix + 101), channels by the cache's order; the drivable
    area on the road_segment channel (the lane channel stays empty: drivable = max of both must read it); in the reference the area beyond our rows 0..98 has no cells"""
    L = np.full((7, REF_N, REF_N), -10.0, np.float32)
    ix, iy = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    ok = ix + 101 < REF_N
    r, c = (151 - iy)[ok], (ix + 101)[ok]
    chan = {"road_segment": "drivable_area", "road_divider": "road_divider", "lane_divider": "lane_divider", "ped_crossing": "ped_crossing", "walkway": "walkway", "carpark_area": "carpark_area"}
    for name, layer in chan.items():
        L[GC_FIXED[name], r, c] = np.where(gt[layer][ok] > 0.5, 10.0, -10.0)
    return L


GC_FIXED = dict(GC)                                         # the true channel order, kept apart from GC so that a mutant of GC does not also change the world it is tested on


def _selftest_checks():
    # 1. the mapping: single cells set in the reference array by hand arrive at the hand-computed cells of our grid, on the right layer
    L = np.full((7, REF_N, REF_N), -10.0, np.float32)
    L[GC_FIXED["lane_divider"], 141, 121] = 10.0                    # row 141 = 151 - iy -> iy 10; column 121 = ix + 101 -> ix 20
    L[GC_FIXED["road_segment"], 100, 150] = 10.0                    # -> iy 51, ix 49: reads as drivable_area (the max of lane and road_segment)
    L[GC_FIXED["lane"], 120, 160] = 10.0                            # -> iy 31, ix 59: also drivable_area
    L[GC_FIXED["road_divider"], 100, 199] = 10.0                    # the last column: iy 51, ix 98 (the last row of the claim region); every row beyond reads it too (clipped)
    m = reference_maps(L)
    for layer, cells in (("lane_divider", {(20, 10)}), ("drivable_area", {(49, 51), (59, 31)}), ("road_divider", {(ix, 51) for ix in range(98, NX)}), ("walkway", set()), ("carpark_area", set()), ("ped_crossing", set())):
        assert {tuple(c) for c in np.argwhere(m[layer] > 0.5)} == cells, (layer, sorted(map(tuple, np.argwhere(m[layer] > 0.5)))[:4])       # rows 99 .. 119 of the last column: the clip artefact of the old region
    # 2. the claim region: 99 rows for the offset 0.5, 100 for 0.0; beyond 50 m refused; the old region keeps the clipped artefact, the claim region does not see it
    assert claim_rows(x_max=50.0) == 99 and claim_rows(0.0, 50.0) == 100 and claim_rows(0.5, 20.0) == 40
    try:
        claim_rows(0.5, 60.0)
        raise AssertionError("x_max 60 m must be refused")
    except ValueError:
        pass
    # 3. the table on the planted world: the reference is right on the GT, ours misses scene 0's and 1's road
    frames, parts, scenes, tokens = _plant_world(), [], [], []
    for k, (sc, gt, fov) in enumerate(frames):
        ours = {n: gt[n].copy() for n in LAYERS}
        if sc < 2:
            ours["drivable_area"] = np.where(np.arange(NY)[None, :] < 52, ours["drivable_area"], 0)      # half of the road is missing in scenes 0 and 1
        ours["walkway"] = gt["walkway"] * 0.4                                                               # right but under-confident: only a threshold of 0.3 sees it
        parts.append(frame_counts({"ours": ours, "ref": reference_maps(_reference_logits(gt))}, gt, fov, claim_rows(x_max=50.0)))
        scenes.append(f"scene-{sc}")
        tokens.append(f"t{k}")
    scores = Scores.stack(tokens, scenes, parts)
    rows = {(r["layer"], r["tol"]): r for r in rows_of(scores, "claim", 200, 0)}
    assert all(abs(rows[(l, t)]["ref"] - 1.0) < 1e-12 for l in LAYERS for t in TOLERANCES if rows[(l, t)]["n_gt"]), "the reference is the GT: IoU 1 on every row with GT cells (orientation, offset, channels)"
    # the independent value of ours on drivable_area iou@0: plain boolean arithmetic over the claim cells
    nr = claim_rows(x_max=50.0)
    tp = fp = fn = 0
    for sc, gt, fov in frames:
        o = (np.where(np.arange(NY)[None, :] < 52, gt["drivable_area"], 0) if sc < 2 else gt["drivable_area"])[:nr] > 0.5
        g = gt["drivable_area"][:nr] > 0.5
        f = fov[:nr]
        tp += (o & g & f).sum()
        fp += (o & ~g & f).sum()
        fn += (~o & g & f).sum()
    want = tp / (tp + fp + fn)
    r = rows[("drivable_area", 0.0)]
    assert abs(r["ours"] - want) < 1e-12 and r["n_gt"] == sum(int(((gt["drivable_area"][:nr] > 0.5) & fov[:nr]).sum()) for _, gt, fov in frames), "ours and n_gt against plain arithmetic"
    assert abs(r["ratio"] - want) < 1e-12 and abs(r["d1"]["diff"] - (want - 1.0)) < 1e-12 and r["d1"]["diff"] < 0, "d1 = ours - reference (the sign), ratio = ours / reference"
    # 4. d2: ours is under-confident on the walkway (0.4): at 0.5 nothing, at its own best threshold 0.3 everything; the reference is best at 0.5 or any (all logits are +-10)
    w = rows[("walkway", 0.0)]
    assert w["ours"] == 0.0 and w["best"]["ours"]["thr"] == 0.3 and abs(w["d2"]["ours"] - 1.0) < 1e-12 and w["d1"]["diff"] == -1.0 and abs(w["d2"]["diff"]) < 1e-12, (w["d1"], w["d2"], w["best"])
    # 5. the interval is the paired scene bootstrap: every draw recomputed from the stored per-frame counts with scene_picks, by plain loops (a bootstrap by frame cannot give this)
    members = {}
    for k, s in enumerate(scenes):
        members.setdefault(s, []).append(k)
    names = sorted(members)
    t05, li = THR.index(0.5), LAYERS.index("drivable_area")
    diffs = []
    for pick in scene_picks(len(names), 200, 0):
        ks = [k for i in pick for k in members[names[i]]]
        diffs.append(iou(scores.cnt[("claim", "ours")][ks, t05, li, 0].sum(0)) - iou(scores.cnt[("claim", "ref")][ks, t05, li, 0].sum(0)))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    assert abs(r["d1"]["lo"] - lo) < 1e-12 and abs(r["d1"]["hi"] - hi) < 1e-12 and r["d1"]["n_valid"] == 200, ((lo, hi), r["d1"])
    # 6. the stored per-frame counts give the same rows (--retable), and the reference-only run has no differences
    again = {(x["layer"], x["tol"]): x for x in rows_of(Scores(scores.tokens, scores.scenes, scores.cnt, scores.n_gt), "claim", 200, 0)}
    assert again[("drivable_area", 0.0)]["d1"] == r["d1"]
    only = Scores(scores.tokens, scores.scenes, {k: v for k, v in scores.cnt.items() if k[1] == "ref"}, scores.n_gt)
    assert "d1" not in rows_of(only, "claim", 10, 0)[0] and "ours" not in rows_of(only, "claim", 10, 0)[0]
    # 7. the old region sees the clipped artefact, the claim region does not: a reference that is right inside the array scores below 1 on the old region at the far rows
    old = {(x["layer"], x["tol"]): x for x in rows_of(scores, "old", 50, 0)}
    assert old[("drivable_area", 0.0)]["ref"] < 1.0 and old[("drivable_area", 0.0)]["n_gt"] > rows[("drivable_area", 0.0)]["n_gt"], "the old region has the cells beyond the array"
    # 8. the claim region is CROPPED, not masked: a GT cell beyond the reference array (row 99) must not confirm a predicted cell inside it through the tolerance dilation
    zero = {k: np.zeros((NX, NY), np.float32) for k in LAYERS}
    pred, gtl = {**zero, "lane_divider": np.zeros((NX, NY), np.float32)}, {**zero, "lane_divider": np.zeros((NX, NY), np.float32)}
    pred["lane_divider"][98, 40], gtl["lane_divider"][99, 40] = 1.0, 1.0
    c, _ = frame_counts({"ref": pred}, gtl, np.ones((NX, NY), bool), claim_rows(x_max=50.0))
    ld = LAYERS.index("lane_divider")
    assert list(c[("claim", "ref")][THR.index(0.5), ld, 1]) == [0, 1, 0, 0] and list(c[("old", "ref")][THR.index(0.5), ld, 1]) == [1, 0, 1, 0], (c[("claim", "ref")][1, ld, 1], c[("old", "ref")][1, ld, 1])
    # 9. the identity of the GT: another one and an UNKNOWN one are refused unless told (no run, a folder that does not exist, no file, no key); the same GT (also through a version that makes it) is not
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        os.makedirs(os.path.join(t, "empty"))
        os.makedirs(os.path.join(t, "nokey"))
        os.makedirs(os.path.join(t, "v8"))
        json.dump({}, open(os.path.join(t, "nokey", "gt.json"), "w"))
        json.dump({"gt_signature": "0.5_4_6a7aa7864a"}, open(os.path.join(t, "v8", "gt.json"), "w"))
        assert trained_gt(None) is None and trained_gt(os.path.join(t, "nowhere")) is None and trained_gt(os.path.join(t, "empty")) is None and trained_gt(os.path.join(t, "nokey")) is None
        assert trained_gt(os.path.join(t, "v8")) == "0.5_4_6a7aa7864a"

        def refused(fn):
            try:
                fn()
            except SystemExit as e:
                return str(e)
            return ""
        for run in (None, os.path.join(t, "nowhere"), os.path.join(t, "empty"), os.path.join(t, "nokey")):
            msg = refused(lambda: gt_identity(run, "6a7aa7864a"))
            assert "unknown" in msg and "--allow-gt-mismatch" in msg, (run, msg)             # no TypeError for run None, no pass for an unknown GT
        assert "trained on 0.5_4_6a7aa7864a" in refused(lambda: gt_identity(os.path.join(t, "v8"), "a107e0b4e8"))
        assert check_gt("a107e0b4e8", "0.5_4_6a7aa7864a", True) is False and check_gt("6a7aa7864a", None, True) is None
        assert check_gt("c643a7539d", "0.5_4_6a7aa7864a") is True and check_gt("6a7aa7864a", "0.5_4_6a7aa7864a") is True, "the same GT through the version that makes it (v9 = v8)"
    # 10. frames of a stored list that the run's val set does not have are reported, not dropped silently
    uni = [{"i": k, "token": f"t{k}", "scene": f"s{k // 2}"} for k in range(6)]
    sel, missing = select_frames(uni, None, ["t3", "tx", "t1", "ty"])
    assert [u["token"] for u in sel] == ["t3", "t1"] and missing == ["tx", "ty"], (sel, missing)
    assert select_frames(uni, 3, None)[1] == [] and len(select_frames(uni, 3, None)[0]) == 3
    # 11. the best-threshold comparison says it is in-sample, the plain one does not
    assert r["d2"]["in_sample_thresholds"] is True and "in_sample_thresholds" not in r["d1"]
    return "soa_maps selftest ok: mapping by hand (orientation, offset, channels), claim region, ours / n_gt / ratio against plain arithmetic, d1 sign, d2 thresholds, interval = the scene bootstrap draw by draw, crop not mask, old region artefact, GT identity (unknown = refused), missing frames reported, in-sample flag"


def selftest():
    return _selftest_checks()


MUTANTS = {}


def mutant(name):
    def deco(fn):
        MUTANTS[name] = fn
        return fn
    return deco


@mutant("orientation: the row follows +y instead of -y")
def _m_orient():
    keep = reference_index

    def bad(offset=None):
        row, col = keep(offset)
        return REF_N - 1 - row, col
    return {"reference_index": bad}


@mutant("forward offset 0 instead of 0.5")
def _m_offset():
    return {"OFFSET": 0.0}


@mutant("channel order: lane_divider and road_divider swapped")
def _m_channels():
    return {"GC": {**GC, "lane_divider": GC["road_divider"], "road_divider": GC["lane_divider"]}}


@mutant("drivable area = the lane channel only (not the max with road_segment)")
def _m_drivable():
    return {"GC": {**GC, "road_segment": GC["lane"]}}


@mutant("bootstrap by frame instead of by scene")
def _m_frame():
    return {"by_scene": lambda vectors, scenes: {k: [v] for k, v in enumerate(vectors)}}


@mutant("the old clip is back: the claim region is all 120 rows")
def _m_clip():
    return {"claim_rows": lambda offset=None, x_max=CLAIM_X_MAX: NX}


@mutant("difference with the other sign (reference - ours)")
def _m_sign():
    return {"bootstrap": lambda ref_by, ours_by, reduce, keys, n, seed: {k: {"a": v["b"], "b": v["a"], "diff": -v["diff"], "lo": -v["hi"], "hi": -v["lo"], "n_valid": v["n_valid"]}
                                                                          for k, v in bootstrap_scenes(ref_by, ours_by, reduce, keys, n=n, seed=seed).items()}}


@mutant("d2 takes threshold 0.5 for everyone")
def _m_best():
    return {"best_thresholds": lambda total: np.full(total.shape[1:3], THR.index(0.5))}


@mutant("the claim region masks the far rows instead of cropping the maps")
def _m_mask():
    keep = frame_counts

    def bad(maps, gt, fov, rows):
        f = np.asarray(fov).copy()
        f[rows:] = False
        return keep(maps, gt, f, NX)                                  # all rows reach the dilation; only the counted cells are cut (and the old region is then cut too)
    return {"frame_counts": bad}


@mutant("the camera cells are not applied (every cell scored)")
def _m_fov():
    keep = frame_counts

    def bad(maps, gt, fov, rows):
        return keep(maps, gt, np.ones_like(np.asarray(fov), bool), rows)
    return {"frame_counts": bad}


@mutant("an unknown training GT passes the identity check")
def _m_unknown():
    keep = check_gt
    return {"check_gt": lambda sig, trained, allow_mismatch=False: None if trained is None else keep(sig, trained, allow_mismatch)}


@mutant("no run folder is a TypeError instead of an unknown GT")
def _m_run_none():
    return {"trained_gt": lambda run: json.load(open(os.path.join(run, "gt.json"))).get("gt_signature")}


@mutant("frames missing from the run's val set are dropped silently")
def _m_silent():
    keep = select_frames
    return {"select_frames": lambda universe, n=None, only=None: (keep(universe, n, only)[0], [])}


def mutants():
    """each planted bug must make the self-test fail; one that does not ESCAPED, the test could not have found that bug"""
    module = sys.modules[__name__]
    escaped = 0
    for name, make in MUTANTS.items():
        patch = make()
        saved = {k: getattr(module, k) for k in patch}
        try:
            for k, v in patch.items():
                setattr(module, k, v)
            try:
                _selftest_checks()
                verdict = "ESCAPED"
                escaped += 1
            except Exception as ex:                                       # noqa: BLE001
                verdict = f"caught ({type(ex).__name__})"
        finally:
            for k, v in saved.items():
                setattr(module, k, v)
        print(f"{verdict:24s} {name}")
    _selftest_checks()                                                    # and the unmutated code passes again
    print(f"{len(MUTANTS) - escaped} of {len(MUTANTS)} planted bugs caught")
    return 1 if escaped else 0


if __name__ == "__main__":
    raise SystemExit(main())
