"""Our map layers against the published camera + radar map model GaussianCaR on the SAME frames, the SAME ground truth (autogt v8, the GT
of the current runs), the SAME mask (the GT camera cells, 'fov') and the SAME metric code (evaluation/metrics.SegMeter), CPU only.

Usage (bev_fusion folder, under nice; the GPU is never touched: CUDA_VISIBLE_DEVICES is emptied before torch is imported):
    nice -n 15 env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/ml/miniconda3/envs/gcar/bin/python temp/t16/soa_maps_v8.py RUN OUT.json [LIMIT]
        [--head] [--stored] [--no-model] [--all-dts] [--rule old|usable|old400] [--prior N] [--offset M] [--align] [--no-check] [--store DIR]
  RUN        run folder, e.g. runs/t16_full_s0 (the checkpoint = the last ckpt_*.pt under its EMA weights, under the run's own code snapshot)
  OUT.json   the numbers; next to it OUT.md (the tables) and OUT_frames.npz (the additive counts of every frame, for a paired bootstrap)
  LIMIT      number of frames, default 400 = all of the base set. Base set (--rule old, the default): the 400 frames of the old scoring
             (temp/soa_gap/map_ref_dev.py: the GaussianCaR logits cached by the lane project, val_tokens.json, 1706 frames in 43 of our 82 val scenes;
             every 4th of them) that are samples of this run's val set: the datasets of the current framework leave out the first and last keyframe
             of every scene, so 384 of the 400 remain (16 frames lost). --rule usable: every 4th of the CURRENT usable list (the old recipe, other
             frames); --rule old400 (with --no-model only): all 400 old frames, edge keyframes included. A smaller LIMIT spreads over the base set.
  --head     the FIRST LIMIT frames of the base set instead of spreading (a LIMIT run is then a prefix of the full run)
  --stored   only frames of the base set that are also in the run's stored 1000-frame suite evaluation (sanity gate: the per-frame counts of our model
             are compared with the stored ones); the check runs for every scored frame that is in that store anyway (--no-check: not)
  --no-model GaussianCaR and the prior only (no network; a couple of minutes for 400 frames): the check of the reference against the old numbers
  --all-dts  the network decodes -0.5 / 0 / +0.5 like the suite (needs the trainer's GT cache entry of every frame); default: dt 0 only, which is the
             same prediction at dt 0 (4 frames, both ways: identical counts) and a third faster
  --prior N  train frames of the per-cell prior map (null), default 500 as the suite's `eval --prior 500`
  --offset M forward offset of the GaussianCaR arrays against our grid in metres, default 0.5 (temp/soa_gap/align_gcar.py)
  --align    also print the drivable IoU of GaussianCaR for offsets 0 ... 1 m on these frames (is the mapping still right on the v8 GT?)

What is scored, per layer (drivable_area, ped_crossing, walkway, carpark_area, lane_divider, road_divider), per range bin (all, 0-20, 20-40, 40-60 m)
and tolerance (iou@0, iou@0.5, iou@1; SegMeter: a predicted cell is right if a GT cell centre lies within the tolerance and vice versa):
  ours      the run's semantic head at dt 0, the memory EMPTY before every frame (the suite's `eval --n`, state reset = the numbers of record)
  gcar      GaussianCaR logits: drivable_area = max(sigmoid(lane), sigmoid(road_segment)), the other five layers directly (it has no stop line);
            mapping of the arrays: row = -y_left, column = forward + OFFSET
  prior     the per-cell mean of the GT over N train keyframes (what the position in the grid alone gives)
  thresholds 0.5 (the evaluation of record) and 0.3 / 0.7; "best of" = the best of the three per layer / bin / tolerance, chosen on these frames
            (IN-SAMPLE, labelled so; it is for the model trained with another loss)
  ours also gets the zero-crossing F1 and field error of the two divider layers (from its distance field; GaussianCaR has none)
GT identity: the dataset's own signature must equal runs/RUN/gt.json (v8: 0.5_4_6a7aa7864a); a mismatch stops the script. The GT of a frame is the
trainer's cache file (what the suite scored against) or, where the cache has none (only 1463 of the 3109 val frames are cached for this signature), the
stored GT data/autogt/c643a7539d/<token>.npz of the same version (v9 code = v8 GT byte for byte, autogt/versions.csv same_gt_as; the script compares
12 frames that have both: every layer equal). A frame with neither, or without its radar cache entry, stops the script: the dataset would build and
WRITE it into data/cache/, and this script writes only OUT*.

Deviations from the old method (temp/soa_gap/map_ref_dev.py): the old training interface (bev_probe, make_datasets from live training/) is replaced by the
suite's own pieces (evaluation/bev_adapter: the run's code snapshot, Source.load, Subject.predict); the nuScenes devkit is the memory-light one of
autogt/light_nusc.py (CAM_FRONT + radars of the 82 val scenes, ~0.7 GB instead of 8+ GB: the box is shared); the thresholds are applied to a binary
prediction (p > thr) instead of shifting the probability; the prior uses the suite's 500 train frames; counts are kept per frame.
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""            # before torch: no GPU, whatever the code does
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "2"
try:
    if os.nice(0) < 15:
        os.nice(15 - os.nice(0))
except OSError:
    pass

import argparse  # noqa: E402
import datetime  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))                 # bev_fusion
WS = os.path.abspath(os.path.join(ROOT, ".."))                         # the workspace: autogt/, autoresearch/
if sys.path and os.path.abspath(sys.path[0]) == HERE:
    sys.path.pop(0)                                                    # the folder of this script holds many scratch files: none of them may shadow a module
for _p in (os.path.join(WS, "autoresearch"), WS, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

LAYERS = ("drivable_area", "ped_crossing", "walkway", "carpark_area", "lane_divider", "road_divider")
LINES = ("lane_divider", "road_divider")
AGG = {"mine": {}, "theirs": {}}                      # summed counts of the frames that are in the stored evaluation: ours (CPU) and the stored ones (GPU)
BINS = ("all", "0-20", "20-40", "40-60")
TOLS = ("iou@0", "iou@0.5", "iou@1")
THR = (0.3, 0.5, 0.7)
SYSTEMS = ("ours", "gcar", "prior")
GC_IDX = {"lane": 0, "road_segment": 1, "road_divider": 2, "lane_divider": 3, "ped_crossing": 4, "walkway": 5, "carpark_area": 6}
HOME = os.path.expanduser("~")


# ------------------------------------------------------------------------------------------------ helpers
def sig(a):
    return 1.0 / (1.0 + np.exp(-np.clip(a.astype(np.float32), -30, 30)))


def gc_index(grid, offset):
    """(row, column) of GaussianCaR's 200 x 200 array for every cell of our grid: row = -y_left, column = forward + offset (0.5 m cells, +-50 m)"""
    xs = grid.x_min + (np.arange(grid.nx) + 0.5) * grid.res
    ys = grid.y_min + (np.arange(grid.ny) + 0.5) * grid.res
    X, Y = np.meshgrid(xs + offset, ys, indexing="ij")
    return (np.clip(np.floor((-Y + 50.0) / grid.res).astype(int), 0, 199), np.clip(np.floor((X + 50.0) / grid.res).astype(int), 0, 199))


def gcar_maps(L, rc):
    """cached logits [7, 200, 200] -> probabilities of our six layers on our grid"""
    R, C = rc
    p = {k: sig(L[i])[R, C] for k, i in GC_IDX.items()}
    p["drivable_area"] = np.maximum(p["lane"], p["road_segment"])
    return {k: p[k] for k in LAYERS}


def merge(total, c):
    """add the counts c (a SegMeter's .c) to the SegMeter total"""
    for k, acc in c.items():
        t = total.c.get(k)
        if t is None:
            total.c[k] = {"cnt": acc["cnt"].copy(), "n_gt": acc["n_gt"], "err": acc["err"], "n_err": acc["n_err"]}
        else:
            t["cnt"] += acc["cnt"]
            t["n_gt"] += acc["n_gt"]
            t["err"] += acc["err"]
            t["n_err"] += acc["n_err"]


def ikey(dt, name, subset, b):
    return f"fov|{float(dt):+.2f}|{name}|{subset}|{b}"        # the key of the suite's stored counts (bev_task._ikey)


def frame_scores(fm):
    """the per-frame numbers the suite stores in frames.jsonl (seg.*), from the one-frame meter of ours at threshold 0.5"""
    r = fm.result()[0.0]

    def g(layer, m):
        return r[layer]["all"]["all"][m]
    return {"seg.drivable_iou": g("drivable_area", "iou@0"), "seg.walkway_iou": g("walkway", "iou@0"),
            "seg.lane_divider_iou": g("lane_divider", "iou@0.5"), "seg.lane_divider_f1": g("lane_divider:zc", "f1@0.5")}


def num(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def fmt(v, d=3):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{d}f}"


def ratio(a, b):
    return None if a is None or b is None or b == 0 or math.isnan(a) or math.isnan(b) else a / b


class Stored:
    """the per-frame counts of a stored suite evaluation (eval/suite/quant/<name>/): frames.jsonl + counts_seg_<chunk>.npz"""

    def __init__(self, d):
        self.dir = d
        self.frames = {}
        for line in open(os.path.join(d, "frames.jsonl")):
            if line.strip():
                fr = json.loads(line)
                self.frames[fr["key"]] = fr
        self.manifest = json.load(open(os.path.join(d, "manifest.json")))
        self._z = {}

    def counts(self, key):
        fr = self.frames[key]
        z = self._z.get(fr["chunk"])
        if z is None:
            z = self._z[fr["chunk"]] = np.load(os.path.join(self.dir, f"counts_seg_{fr['chunk']:05d}.npz"))
        p = fr["pos"]
        keys, cnt, n_gt, err = z[f"{p}|keys"], z[f"{p}|cnt"], z[f"{p}|n_gt"], z[f"{p}|err"]
        return {str(k): (cnt[j], int(n_gt[j]), float(err[j])) for j, k in enumerate(keys) if str(k).startswith("fov|+0.00|")}


def find_store(run, explicit=None):
    """the run's stored suite evaluation to check against: cold (state reset), val, the largest of the plain eval_*_n* directories"""
    cands = [explicit] if explicit else sorted(glob.glob(os.path.join(run, "eval", "suite", "quant", "eval*_n*")))
    best = None
    for d in cands:
        name = os.path.basename(os.path.normpath(d))
        if not explicit and any(s in name for s in ("_train", "_warm", "_ctl_")):
            continue
        try:
            st = Stored(d)
        except (OSError, ValueError):
            continue
        if st.manifest.get("state") != "reset" or st.manifest.get("source", {}).get("split") != "val":
            continue
        if best is None or len(st.frames) > len(best.frames):
            best = st
    return best


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("limit", nargs="?", type=int, default=400)
    ap.add_argument("--head", action="store_true")
    ap.add_argument("--stored", action="store_true")
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--all-dts", action="store_true")
    ap.add_argument("--rule", choices=("old", "usable", "old400"), default="old")
    ap.add_argument("--prior", type=int, default=500)
    ap.add_argument("--offset", type=float, default=0.5)
    ap.add_argument("--align", action="store_true")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--store", default=None)
    a = ap.parse_args()

    t_start = time.time()
    os.chdir(ROOT)
    run = os.path.relpath(os.path.abspath(a.run), ROOT).replace("\\", "/")
    out_path = os.path.abspath(a.out)
    stem = out_path[:-5] if out_path.endswith(".json") else out_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    load0 = os.getloadavg()
    print(f"soa_maps_v8: {run}, limit {a.limit}, threads 2, nice {os.nice(0)}, load {load0[0]:.1f}, cuda visible: {torch.cuda.is_available()}", flush=True)

    from evaluation import bev_adapter as A  # noqa: E402  (numpy only at import)
    gt_run = json.load(open(os.path.join(run, "gt.json")))
    code = A.setup_paths(run)                                   # the run's own code snapshot first on sys.path, BEFORE anything of the chain is imported
    ns = A._chain()
    ns.code = code
    T = ns.T
    cfg = T.TrainConfig(**json.load(open(os.path.join(run, "config.json"))))
    A.freeze_scenes(cfg)
    cfg.workers, cfg.seq_len, cfg.batch_size = 0, 1, 1
    if not cfg.radar_cache and os.path.isdir(os.path.join(ROOT, "data", "cache", "radar")):
        cfg.radar_cache = "data/cache/radar"
    val_names = list(cfg.val_scenes or [])
    if not val_names or not cfg.feature_cache:
        sys.exit("this script needs a run with a val scene list and a feature cache (a run of the five-layer framework)")

    # the memory-light devkit (CAM_FRONT + radars of the val scenes): make_datasets of the run's snapshot builds its datasets on it unchanged
    import nuscenes.nuscenes as NN
    from autogt.light_nusc import light_nuscenes
    Real = NN.NuScenes

    def light(version, dataroot=None, verbose=False, **kw):
        NN.NuScenes = Real                                      # light_nuscenes subclasses the real class
        try:
            return light_nuscenes(dataroot, version, scenes=val_names)
        finally:
            NN.NuScenes = light
    NN.NuScenes = light
    try:
        train_ds, val_ds = T.make_datasets(cfg)
    finally:
        NN.NuScenes = Real
    del train_ds                                                # empty (the light devkit holds the val scenes only); the prior reads the train GT from the cache
    sig_now = A.gt_signature(val_ds)
    print(f"dataset: {len(val_ds)} val samples, GT signature {sig_now} ({A.gt_version(sig_now)}), the run trained on {gt_run.get('gt_signature')} ({gt_run.get('gt_version')})", flush=True)
    if sig_now != gt_run.get("gt_signature"):
        sys.exit(f"STOP: the GT signature of the dataset ({sig_now}) is not the run's ({gt_run.get('gt_signature')})")
    grid = val_ds.grid
    nusc = val_ds.nusc

    # ---- the frames
    tok = json.load(open(f"{HOME}/exp_gcar_b123/feats/val_tokens.json"))
    pos = {t: i for i, t in enumerate(val_ds.samples)}
    rows = [(i, t) for i, t in enumerate(tok) if t in pos]
    old_idx = list(json.load(open(os.path.join(ROOT, "temp", "soa_gap", "gcar_val_idx_in_our_val.json"))))
    same_usable = [i for i, _ in rows] == old_idx
    # the old scoring: its usable list (1706 cached GaussianCaR frames, then every 4th, 400). The datasets of the current framework leave out the first
    # and the last keyframe of every scene (a sample needs a previous and a next keyframe: 2 x 43 scenes = 86 of the 1706), so of the old 400 the frames
    # still in the val set are kept; --rule usable instead takes every 4th of the CURRENT usable list (other frames, same recipe)
    old_rows = [(i, tok[i]) for i in old_idx]
    old400 = old_rows[:: max(1, len(old_rows) // 400)][:400]
    if a.rule == "old400":                                      # all 400 old frames, edge keyframes included: only without the network (the val dataset has no sample for them)
        if not a.no_model:
            sys.exit("--rule old400 needs --no-model (the dataset of the run has no sample for the first / last keyframe of a scene)")
        base = list(old400)
        rule_text = "all 400 frames of the old scoring (the first / last keyframe of a scene included)"
    elif a.rule == "old":
        base = [r for r in old400 if r[1] in pos]
        rule_text = f"the {len(base)} of the 400 frames of the old scoring that are samples of this val set ({len(old400) - len(base)} are the first / last keyframe of a scene)"
    else:
        base = rows[:: max(1, len(rows) // 400)][:400]
        rule_text = f"every {max(1, len(rows) // 400)}th of the {len(rows)} usable frames ({len(base)} frames; not the same frames as the old 400)"
    print(f"{len(tok)} cached GaussianCaR frames; {len(rows)} are samples of this val set (old scoring: {len(old_idx)}, same list: {same_usable}); base set: {rule_text}", flush=True)
    store = None if a.no_check else find_store(run, a.store)
    if a.stored and store is None:
        sys.exit("--stored: no stored cold val evaluation found for this run")
    if a.stored:
        sel = [r for r in base if r[1] in store.frames][:a.limit]
        how = f"the first {len(sel)} of the base set that are in the stored evaluation"
    elif a.head:
        sel = base[:a.limit]
        how = f"the first {len(sel)} of the base set"
    elif a.limit >= len(base):
        sel = base
        how = rule_text
    else:
        sel = base[:: max(1, len(base) // a.limit)][:a.limit]
        how = f"{len(sel)} frames spread over the base set (step {max(1, len(base) // a.limit)})"
    scene_of = {}
    for t in (t for _, t in sel):
        scene_of[t] = nusc.get("scene", nusc.get("sample", t)["scene_token"])["name"]
    print(f"{len(sel)} frames: {how}; {len(set(scene_of.values()))} scenes; stored evaluation to check against: "
          f"{os.path.basename(store.dir) + ' (' + str(len(store.frames)) + ' frames)' if store else 'none'}", flush=True)

    # ---- the GT: the trainer's cache file of the frame (what the suite scored against) or, where the cache has none, the stored GT of the same version
    # (data/autogt/<canonical signature>/<token>.npz: v9 = v8 byte for byte, versions.csv same_gt_as). Nothing is generated or written: a frame with
    # neither, or without its radar cache entry (the dataset would build and WRITE it into data/cache/), stops the script before the model is loaded.
    from autogt.versions import canonical
    store_dir = os.path.join(WS, "data", "autogt", canonical(sig_now.rsplit("_", 1)[-1]))

    def cache_path(t):
        return os.path.join(val_ds.cache_dir, f"{t}_{sig_now}.npz")

    def read_cache(t):
        with np.load(cache_path(t), allow_pickle=True) as z:
            return z["gt"].item()

    def read_store(t):
        with np.load(os.path.join(store_dir, t + ".npz"), allow_pickle=True) as z:
            return {k: (z[k].item() if k == "meta" else z[k]) for k in z.files}

    def gt_cur(t):
        """(the 'cur' GT dict of a keyframe, where it came from)"""
        if os.path.exists(cache_path(t)):
            return read_cache(t)["cur"], "cache"
        return read_store(t), "store"
    n_src = {"cache": 0, "store": 0}
    miss = {"gt": []}
    n_radar = {"cache": 0, "computed_not_written": 0}
    for _, t in sel:
        if os.path.exists(cache_path(t)):
            n_src["cache"] += 1
        elif os.path.exists(os.path.join(store_dir, t + ".npz")) and not a.all_dts:
            n_src["store"] += 1
        else:
            miss["gt"].append(t)
        if not a.no_model:
            n_radar["cache" if os.path.exists(val_ds._radar_path(nusc.get("sample", t), val_ds.gt.cam(t))) else "computed_not_written"] += 1
    if miss["gt"]:
        sys.exit(f"STOP: no GT (cache{'' if a.all_dts else ' or store'}) for {len(miss['gt'])} frames; first: {miss['gt'][:3]}")
    print(f"GT of the {len(sel)} frames: {n_src['cache']} from the trainer's cache, {n_src['store']} from the stored GT {os.path.basename(store_dir)}", flush=True)
    if not a.no_model:
        print(f"radar input of the {len(sel)} frames: {n_radar['cache']} from the radar cache of the run's code (signature {val_ds._rsig}), {n_radar['computed_not_written']} "
              "not in the cache: computed from the raw sweeps in memory (what the cache would hold), never written", flush=True)
    same_gt = {"checked": 0, "equal": 0, "differ": []}                    # cache file against stored file of the same token: must be the same arrays
    for _, t in [r for r in sel if os.path.exists(cache_path(r[1])) and os.path.exists(os.path.join(store_dir, r[1] + ".npz"))][:12]:
        c, s_ = read_cache(t)["cur"], read_store(t)
        bad = [k for k in c if isinstance(c[k], np.ndarray) and (k not in s_ or c[k].shape != s_[k].shape or not np.array_equal(c[k], s_[k]))]
        same_gt["checked"] += 1
        same_gt["equal"] += not bad
        if bad:
            same_gt["differ"].append({"token": t, "layers": bad[:5]})
    print(f"cache file against stored file: {same_gt['equal']} of {same_gt['checked']} frames equal in every layer", flush=True)
    orig_gt = val_ds._gt
    FRD = type(val_ds).__mro__[1]                                        # FrontRadarDataset: its _radar is what fills the cache
    assert FRD.__name__ == "FrontRadarDataset", FRD
    orig_radar = val_ds._radar

    def radar_no_write(sample, cur_sd):
        """the cached entry, else the same function the cache is filled by, without writing the entry"""
        if os.path.exists(val_ds._radar_path(sample, cur_sd)):
            return orig_radar(sample, cur_sd)
        return FRD._radar(val_ds, sample, cur_sd)
    val_ds._radar = radar_no_write
    radar_same = {"checked": 0, "equal": 0}                              # the live computation against the cache entry of frames that have both
    if not a.no_model:
        for _, t in [r for r in sel if os.path.exists(val_ds._radar_path(nusc.get("sample", r[1]), val_ds.gt.cam(r[1])))][:4]:
            smp, cur_ = nusc.get("sample", t), val_ds.gt.cam(t)
            c, l_ = orig_radar(smp, cur_), FRD._radar(val_ds, smp, cur_)
            radar_same["checked"] += 1
            radar_same["equal"] += int(set(c) == set(l_) and all(torch.equal(c[k], l_[k]) for k in c))
        print(f"radar: live computation against the cache entry: {radar_same['equal']} of {radar_same['checked']} frames equal", flush=True)

    def gt_dataset(token):
        """what the dataset hands the network as GT of a frame: dt 0 only unless --all-dts (the dt 0 prediction does not depend on the other targets)"""
        if os.path.exists(cache_path(token)):
            full = orig_gt(token)
            return full if a.all_dts else {"cur": full["cur"]}
        if a.all_dts:
            raise RuntimeError(f"--all-dts needs the trainer's GT cache entry of {token}")
        return {"cur": read_store(token)}
    val_ds._gt = gt_dataset

    # ---- the network
    src = subject = None
    spec = None
    if not a.no_model:
        src = A._make_source_class()(cfg, val_ds, ns, workers=0, control="", split="val", evaluator=None)
        subject = A._make_subject_classes()[0](run, "cpu", ns, None)
        spec = subject.model.spec
        assert subject.device == "cpu", subject.device
        ref_of = {r.key: r for sc in src.scenes() for r in src.frames(sc)}
        key_of = {name: key for name, (_, kind, key) in A.layer_index(spec).items()}
        print(f"model: {subject.describe()['checkpoint']} step {subject.step}, EMA, device {subject.device}; layers {[k for k in LAYERS if k in key_of]}", flush=True)
        for n in LAYERS:
            assert n in key_of and key_of[n] == n, (n, key_of.get(n))
    gt_key = {n: n for n in LAYERS}

    # ---- GaussianCaR logits, prior
    lg = np.load(f"{HOME}/exp_gcar_b123/feats/val_logit7.npy", mmap_mode="r")
    rc = gc_index(grid, a.offset)
    rcs = {o: gc_index(grid, o) for o in (0.0, 0.25, 0.5, 0.75, 1.0)} if a.align else {}
    align_acc = {(lay, o): [0, 0] for lay in ("drivable_area", "lane_divider") for o in rcs}
    prior = prior_maps(cfg, gt_cur, a.prior)
    extra = {"gt_sources": n_src, "gt_store": os.path.relpath(store_dir, WS).replace("\\", "/"), "radar_sources": n_radar, "radar_live_vs_cache": radar_same, "cache_vs_store": same_gt, "base_rule": rule_text,
             "dts_decoded": "-0.5 / 0 / +0.5 (the suite's), scored at 0" if a.all_dts else "0 only (the dt 0 prediction does not depend on the other dts)"}

    # ---- the loop
    from evaluation.metrics import SegMeter  # noqa: E402
    total = {(s, t): SegMeter(grid) for s in SYSTEMS for t in THR}
    perframe = {(s, t): [] for s in SYSTEMS for t in THR}
    frames_info, checks = [], []
    t_model = t_metric = 0.0
    done = 0
    t0 = time.time()
    try:
        for n, (i, t) in enumerate(sel):
            tm = time.time()
            if subject is not None:
                sample = src.load(ref_of[t])
                g = sample.reference[0.0]
                subject.reset()                                      # the suite's state "reset": an empty memory before every frame
                pred = subject.predict(sample, ())
                o = pred.outputs["dt"][0.0]
                ours, ours_dist = {k: o["prob"][k] for k in LAYERS}, o["dist"]
            else:
                g = gt_cur(t)[0]
                ours = ours_dist = None
            t_model += time.time() - tm
            tm = time.time()
            fov = np.asarray(g["camera"], np.float32) > 0
            gt = {k: np.asarray(g[gt_key[k]], np.float32) for k in LAYERS}
            L = np.asarray(lg[i])
            preds = {"gcar": gcar_maps(L, rc), "prior": prior}
            if ours is not None:
                preds["ours"] = ours
            if a.align:                                              # plain IoU of two layers by offset: the thin lane divider is the sensitive one
                pl = {"drivable_area": np.maximum(sig(L[0]), sig(L[1])), "lane_divider": sig(L[3])}
                for o_, (R_, C_) in rcs.items():
                    for lay, p in pl.items():
                        q, d = p[R_, C_] > 0.5, gt[lay] > 0.5
                        align_acc[(lay, o_)][0] += int((q & d & fov).sum())
                        align_acc[(lay, o_)][1] += int(((q | d) & fov).sum())
            fms = {}
            for s, pm in preds.items():
                for th in THR:
                    fm = SegMeter(grid)
                    fm.add({k: (pm[k] > th).astype(np.float32) for k in LAYERS}, gt, mask=fov)
                    if s == "ours" and th == 0.5:
                        dd = {k: ours_dist[k] for k in LINES if k in ours_dist}
                        fm.add_lines(dd, {k: np.asarray(g[k + "_dist"], np.float32) for k in dd}, mask=fov)
                    fms[(s, th)] = fm
                    merge(total[(s, th)], fm.c)
                    perframe[(s, th)].append({k: (v["cnt"].copy(), v["n_gt"], v["err"], v["n_err"]) for k, v in fm.c.items()})
            t_metric += time.time() - tm
            info = {"i": i, "token": t, "scene": scene_of[t]}
            if store is not None and t in store.frames and ("ours", 0.5) in fms:
                sc = store.counts(t)
                fm = fms[("ours", 0.5)]
                dmax = emax = 0.0
                bad_keys = []
                nmiss = 0
                for (dt_, name, sub, b), acc in fm.c.items():
                    sk = ikey(dt_, name, sub, b)
                    if sk not in sc:
                        nmiss += 1
                        continue
                    dk = max(float(np.abs(acc["cnt"] - sc[sk][0]).max()), abs(acc["n_gt"] - sc[sk][1]))
                    dmax = max(dmax, dk)
                    if dk > 0:
                        bad_keys.append([sk, dk])
                    emax = max(emax, abs(acc["err"] - sc[sk][2]))
                    AGG["mine"][sk] = AGG["mine"].get(sk, 0) + acc["cnt"]
                    AGG["theirs"][sk] = AGG["theirs"].get(sk, 0) + sc[sk][0]
                mine = frame_scores(fm)
                stored_scores = store.frames[t]["scores"]
                sdiff, nan_ok = 0.0, True
                for k_, v in mine.items():
                    w = stored_scores.get(k_)
                    if w is None:
                        continue
                    if (v is None or v != v) != (w is None or w != w):
                        nan_ok = False
                    elif v == v and w == w:
                        sdiff = max(sdiff, abs(v - w))
                checks.append({"token": t, "label": store.frames[t]["label"], "max_count_diff": dmax, "keys_differing": len(bad_keys), "differing": bad_keys[:12], "keys_compared": len(fm.c) - nmiss, "max_err_diff": emax, "keys_not_in_store": nmiss,
                               "max_score_diff": sdiff, "nan_pattern_equal": nan_ok, "scores": {k_: num(v) for k_, v in mine.items()}})
                near = {k: int(((np.abs(ours[k] - 0.5) < 1e-4) & fov).sum()) for k in LAYERS}              # cells whose probability sits on the threshold: a float tie between CPU and GPU
                dd = np.where(fov, np.abs(ours["drivable_area"] - 0.5), 9.0)
                ix, iy = np.unravel_index(int(dd.argmin()), dd.shape)
                checks[-1].update({"cells_within_1e-4_of_threshold": near, "closest_drivable_cell": [int(ix), int(iy), float(ours["drivable_area"][ix, iy])]})
                info["checked"] = True
            frames_info.append(info)
            done += 1
            if done % 5 == 0 or done == len(sel) or done <= 2:
                el = time.time() - t0
                print(f"{done}/{len(sel)} frames, {el / done:.1f} s per frame (model {t_model / done:.1f}, metrics {t_metric / done:.1f}), eta {el / done * (len(sel) - done):.0f} s", flush=True)
    except BaseException:
        finish(a, run, out_path, stem, cfg, subject, sig_now, gt_run, sel, frames_info, total, perframe, checks, t_model, t_metric, done, t0, load0,
               same_usable, how, store, align_acc, extra, partial=True)
        raise
    finish(a, run, out_path, stem, cfg, subject, sig_now, gt_run, sel, frames_info, total, perframe, checks, t_model, t_metric, done, t0, load0,
           same_usable, how, store, align_acc, extra, partial=False)
    print(f"total {time.time() - t_start:.0f} s", flush=True)


def prior_maps(cfg, gt_cur, n):
    """per-cell mean of the GT layers over n train keyframes spread evenly (the suite's prior baseline: bev_adapter.prior_maps on the train dataset);
    the train sample list is read from sample.json / scene.json (same order and rule as the dataset), the GT from gt_cur (trainer cache, else the stored GT;
    a keyframe with neither is skipped and counted)"""
    root = os.path.join(os.path.expanduser(cfg.data_root), cfg.version)
    scenes = {s["token"]: s["name"] for s in json.load(open(os.path.join(root, "scene.json")))}
    names = set(cfg.train_scenes or [])
    toks = [s["token"] for s in json.load(open(os.path.join(root, "sample.json")))
            if (not names or scenes[s["scene_token"]] in names) and s["prev"] and s["next"]]
    idx = np.linspace(0, len(toks) - 1, n).round().astype(int)
    acc = {k: 0.0 for k in LAYERS}
    used = skipped = 0
    for j in idx:
        try:
            g = gt_cur(toks[j])[0]
        except FileNotFoundError:
            skipped += 1
            continue
        for k in LAYERS:
            acc[k] = acc[k] + np.asarray(g[k], np.float64)
        used += 1
    print(f"prior: {used} train frames of {len(toks)} train samples (n {n}, {skipped} without GT)", flush=True)
    if used == 0:
        sys.exit("prior: no train GT found")
    prior_maps.info = {"train_samples": len(toks), "used": used, "skipped": skipped}
    return {k: (v / used).astype(np.float32) for k, v in acc.items()}


# ------------------------------------------------------------------------------------------------ output
def finish(a, run, out_path, stem, cfg, subject, sig_now, gt_run, sel, frames_info, total, perframe, checks, t_model, t_metric, done, t0, load0,
           same_usable, how, store, align_acc, extra, partial):
    if done == 0:
        return
    wall = time.time() - t0
    S = {}
    ZC = {}
    for s in SYSTEMS:
        if s == "ours" and subject is None:
            continue
        S[s] = {}
        for th in THR:
            res = total[(s, th)].result()[0.0]
            for layer, subsets in res.items():
                if layer.endswith(":zc"):
                    if th == 0.5:
                        ZC.setdefault(layer, {})
                        for b, row in subsets["all"].items():
                            ZC[layer][b] = {k: num(v) for k, v in row.items()}
                    continue
                for b, row in subsets["all"].items():
                    S[s].setdefault(layer, {}).setdefault(b, {})[str(th)] = {**{k: num(row[k]) for k in TOLS}, "n_gt": row["n_gt"]}
    scenes = sorted({f["scene"] for f in frames_info})
    gate = None
    if checks:
        gate = {"n_frames": len(checks), "store": os.path.basename(store.dir), "max_count_diff": max(c["max_count_diff"] for c in checks),
                "max_err_diff": max(c["max_err_diff"] for c in checks), "max_score_diff": max(c["max_score_diff"] for c in checks),
                "nan_pattern_equal": all(c["nan_pattern_equal"] for c in checks),
                "keys_compared": sum(c["keys_compared"] for c in checks), "keys_differing": sum(c["keys_differing"] for c in checks),
                "headline_from_summed_counts": agg_headline(), "frames": checks}
    old = old_comparison(S) if a.rule == "old400" and len(frames_info) == 400 else None
    meta = {"script": "temp/t16/soa_maps_v8.py", "command": " ".join(sys.argv), "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "partial": partial, "run": run, "step": subject.step if subject else None, "checkpoint": subject.describe()["checkpoint"] if subject else None,
            "weights": "EMA", "gt_signature": sig_now, "gt_version": gt_run.get("gt_version"), "gt_signature_run": gt_run.get("gt_signature"),
            "split": "val (the 82 official val scenes of the run's config; GaussianCaR logits exist for 43 of them)",
            "n_frames": done, "n_scenes": len(scenes), "frames_rule": how, "same_usable_list_as_old_scoring": same_usable,
            "mask": "fov = the GT camera cells (g['camera'] > 0), as the table of record's seg_fov",
            "memory": "ours: empty before every frame (Subject.reset(), the suite's state reset); GaussianCaR: single frame by construction",
            **extra,
            "thresholds": list(THR), "best_of_note": "best of the thresholds per layer / bin / tolerance chosen on these frames: IN-SAMPLE",
            "gcar_offset_m": a.offset, "prior": getattr(prior_maps, "info", None),
            "seconds_per_frame": wall / done, "seconds_model": t_model / done, "seconds_metrics": t_metric / done, "threads": 2, "device": "cpu",
            "load_start": load0[0], "load_end": os.getloadavg()[0]}
    if align_acc and any(v[1] for v in align_acc.values()):
        meta["gcar_iou_by_offset_m"] = {f"{lay}/{o}": v[0] / max(1, v[1]) for (lay, o), v in align_acc.items()}
    summary = {"meta": meta, "frames": frames_info, "systems": S, "ours_zero_crossing_thr0.5": ZC, "stored_check": gate, "vs_old_scoring": old}
    path = out_path + ".partial" if partial else out_path
    json.dump(summary, open(path, "w"), indent=1)
    np.savez_compressed(stem + ("_frames.npz.partial" if partial else "_frames.npz"), **frame_arrays(perframe))
    md = markdown(summary, run)
    open(stem + (".md.partial" if partial else ".md"), "w").write(md)
    print(f"wrote {path} (+ .md, _frames.npz)", flush=True)
    if gate:
        print(f"stored-count gate: {gate['n_frames']} frames, max count diff {gate['max_count_diff']:.3g}, max zc error-sum diff {gate['max_err_diff']:.3g}, "
              f"max per-frame score diff {gate['max_score_diff']:.3g}, NaN pattern equal {gate['nan_pattern_equal']}", flush=True)


def frame_arrays(perframe):
    out = {}
    for (s, th), lst in perframe.items():
        if not lst:
            continue
        for k in lst[0]:
            nm = f"{s}|{th}|{float(k[0]):+.2f}|{k[1]}|{k[2]}|{k[3]}"
            out[nm + "|cnt"] = np.stack([f[k][0] for f in lst])
            out[nm + "|n_gt"] = np.array([f[k][1] for f in lst])
            if k[1].endswith(":zc"):
                out[nm + "|err"] = np.array([f[k][2] for f in lst])
                out[nm + "|n_err"] = np.array([f[k][3] for f in lst])
    return out


def agg_headline():
    """headline numbers of ours from the summed counts of the checked frames: CPU (this script) / the stored GPU counts of the suite / difference"""
    def val(c, t, zc):
        ph, pm, gh, gm = c[t]
        if zc:
            p, r = ph / (ph + pm), gh / (gh + gm)
            return 2 * p * r / (p + r)
        tp = (ph + gh) / 2
        return tp / (tp + pm + gm)
    out = {}
    for layer, t, zc, name in (("drivable_area", 0, False, "drivable_area iou@0"), ("ped_crossing", 0, False, "ped_crossing iou@0"), ("walkway", 0, False, "walkway iou@0"),
                               ("carpark_area", 0, False, "carpark_area iou@0"), ("lane_divider", 1, False, "lane_divider iou@0.5"),
                               ("road_divider", 2, False, "road_divider iou@1"), ("lane_divider:zc", 1, True, "lane_divider:zc f1@0.5"), ("road_divider:zc", 1, True, "road_divider:zc f1@0.5")):
        k = f"fov|+0.00|{layer}|all|all"
        if k in AGG["mine"]:
            try:
                m_, s_ = float(val(AGG["mine"][k], t, zc)), float(val(AGG["theirs"][k], t, zc))
            except ZeroDivisionError:
                continue
            out[name] = {"cpu": m_, "stored_gpu": s_, "diff": m_ - s_}
    return out


def best_of(S, s, layer, b, tol):
    vals = [(S[s][layer][b][str(th)][tol], th) for th in THR]
    vals = [v for v in vals if v[0] is not None]
    return max(vals) if vals else (None, None)


def old_comparison(S):
    """the GaussianCaR (and prior) numbers of the old scoring on the same 400 frames (old GT) next to these ones (v8)"""
    p = os.path.join(ROOT, "temp", "soa_gap", "map_ref_r4soa_400.json")
    if not os.path.exists(p):
        return None
    o = json.load(open(p))["systems"]
    out = {"file": "temp/soa_gap/map_ref_r4soa_400.json (GT of 2026-10-30)"}
    for s in ("gcar", "prior"):
        out[s] = {}
        for layer in LAYERS:
            for tol in TOLS:
                new_ = S[s][layer]["all"]["0.5"][tol]
                old_ = o[s][layer]["all"]["0.5"][tol]
                out[s][f"{layer}/{tol}"] = {"old": old_, "v8": new_, "diff": None if new_ is None or old_ is None else new_ - old_,
                                            "n_gt_old": o[s][layer]["all"]["0.5"]["n_gt"], "n_gt_v8": S[s][layer]["all"]["0.5"]["n_gt"]}
    return out


def markdown(R, run):
    S, M = R["systems"], R["meta"]
    have = "ours" in S
    rec = {}
    rp = os.path.join(ROOT, run, "eval", "by_n", "metrics_n1000.json")
    if os.path.exists(rp):
        rec = json.load(open(rp)).get("metrics", {})

    def rv(s, layer, b, tol):
        return rec.get(f"{'seg_fov' if s == 'ours' else 'prior_fov'}/seg/dt+0.00/{layer}/all/{b}/{tol}")
    L = []
    w = L.append
    who = f"{run} (ours)" if have else "no model (GaussianCaR and the prior only)"
    w(f"# Map layers: {who} against GaussianCaR, GT {M['gt_version']} ({M['gt_signature']}), VAL")
    w("")
    w(f"{M['n_frames']} val frames of {M['n_scenes']} scenes ({M['frames_rule']}), camera cells only (the GT camera mask, as `seg_fov`), "
      + (f"checkpoint {M['checkpoint']} step {M['step']} (EMA), " if have else "")
      + "memory empty before every frame, dt 0. VAL, not TEST. The denominator of each number is "
      "the number of GT cells (`n_gt`, summed over the frames) in the table. GaussianCaR: 6 cameras + radar, 360 degrees, trained for the nuScenes maps "
      f"(its cached logits, forward offset {M['gcar_offset_m']} m); ours: front camera + radar. Prior: per-cell GT mean of {(M['prior'] or {}).get('used')} train frames (null). "
      "Tolerance t: a cell counts as right if the other map has a cell within t metres (iou@0 = plain IoU)."
      + (" **Partial run.**" if M["partial"] else ""))
    w("")

    def table(title, tols, bin_="all", best=False):
        w(f"## {title}")
        w("")
        head = "| layer | metric | n_gt cells | prior | ours | GaussianCaR | ours / GaussianCaR |" + (" ours (record, 1000 frames, 82 scenes) |" if rec and not best else "")
        w(head)
        w("|" + "---|" * (head.count("|") - 1))
        for layer in LAYERS:
            for tol in tols:
                n_gt = S["gcar"][layer][bin_]["0.5"]["n_gt"]
                if n_gt == 0:
                    w(f"| {layer} | {tol} | 0 | - | - | - | - |" + (" - |" if rec and not best else ""))
                    continue
                if best:
                    pr, _ = best_of(S, "prior", layer, bin_, tol)
                    ou, to = best_of(S, "ours", layer, bin_, tol) if have else (None, None)
                    gc, tg = best_of(S, "gcar", layer, bin_, tol)
                    w(f"| {layer} | {tol} | {n_gt} | {fmt(pr)} | {fmt(ou)}{f' (thr {to})' if have else ''} | {fmt(gc)} (thr {tg}) | {fmt(ratio(ou, gc), 2)} |")
                else:
                    pr, gc = S["prior"][layer][bin_]["0.5"][tol], S["gcar"][layer][bin_]["0.5"][tol]
                    ou = S["ours"][layer][bin_]["0.5"][tol] if have else None
                    r_ = rv("ours", layer, bin_, tol)
                    w(f"| {layer} | {tol} | {n_gt} | {fmt(pr)} | {fmt(ou)} | {fmt(gc)} | {fmt(ratio(ou, gc), 2)} |" + (f" {fmt(r_)} |" if rec else ""))
        w("")
    table("Table 1. Prediction threshold 0.5, all ranges (the numbers to quote)", TOLS)
    table("Table 2. Best of the thresholds {0.3, 0.5, 0.7} per layer and metric, chosen on these frames (IN-SAMPLE, for the model trained with another loss)", TOLS, best=True)
    w("## Table 3. By range, threshold 0.5 (ours / GaussianCaR / ratio; n_gt of the bin in brackets)")
    w("")
    w("| layer | metric | 0-20 m | 20-40 m | 40-60 m |")
    w("|---|---|---|---|---|")
    for layer in LAYERS:
        for tol in ("iou@0", "iou@0.5"):
            cells = []
            for b in BINS[1:]:
                ou = S["ours"][layer][b]["0.5"][tol] if have else None
                gc = S["gcar"][layer][b]["0.5"][tol]
                n_b = S["gcar"][layer][b]["0.5"]["n_gt"]
                cells.append(f"{fmt(ou)} / {fmt(gc)} / {fmt(ratio(ou, gc), 2)} ({n_b})" if n_b else "- (no GT cells)")
            w(f"| {layer} | {tol} | " + " | ".join(cells) + " |")
    w("")
    if have and R.get("ours_zero_crossing_thr0.5"):
        w("## Table 4. Ours, divider layers from the distance field (zero crossing; GaussianCaR has no distance field)")
        w("")
        w("| layer | f1@0 | f1@0.5 | f1@1 | precision@0.5 | recall@0.5 | field_mae (m) |")
        w("|---|---|---|---|---|---|---|")
        for layer, d in R["ours_zero_crossing_thr0.5"].items():
            r_ = d["all"]
            w(f"| {layer} | {fmt(r_['f1@0'])} | {fmt(r_['f1@0.5'])} | {fmt(r_['f1@1'])} | {fmt(r_['p@0.5'])} | {fmt(r_['r@0.5'])} | {fmt(r_['field_mae'])} |")
        w("")
    if R.get("stored_check"):
        g = R["stored_check"]
        w(f"Sanity (ours against the suite's stored numbers): {g['n_frames']} of these frames are in the stored suite evaluation `{g['store']}` (made on the GPU). "
          f"{g['keys_compared'] - g['keys_differing']} of {g['keys_compared']} per-frame count arrays are identical; the others differ by at most {g['max_count_diff']:.3g} cells "
          f"(a cell whose probability lies on the 0.5 threshold, CPU against GPU floats); largest per-frame score difference {g['max_score_diff']:.2g}, NaN pattern equal: {g['nan_pattern_equal']}. "
          "From the summed counts (CPU / stored / difference): "
          + "; ".join(f"{k} {v['cpu']:.4f} / {v['stored_gpu']:.4f} / {v['diff']:+.5f}" for k, v in g["headline_from_summed_counts"].items()) + ".")
        w("")
    if R.get("vs_old_scoring"):
        w("Sanity: GaussianCaR on the old GT (2026-10-30) against v8 on the same 400 frames, threshold 0.5, iou@0 / iou@0.5 / iou@1 -> "
          + "; ".join(f"{k.split('/')[0]} {v['old']:.3f}->{v['v8']:.3f}" for k, v in R["vs_old_scoring"]["gcar"].items() if k.endswith("iou@0.5")))
        w("")
    if M.get("gcar_iou_by_offset_m"):
        w("GaussianCaR plain IoU (thr 0.5, camera cells) by forward offset in m: " + ", ".join(f"{k} {v:.3f}" for k, v in M["gcar_iou_by_offset_m"].items()))
        w("")
    w(f"Time: {M['seconds_per_frame']:.1f} s per frame on {M['device']} with {M['threads']} threads (model {M['seconds_model']:.1f} s, metrics {M['seconds_metrics']:.1f} s; "
      f"load average {M['load_start']:.1f} at the start, {M['load_end']:.1f} at the end); command: `{M['command']}`.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
