"""Command line of the BEV system's evaluation: qualitative and quantitative suite on a training run (evalsuite + bev_adapter).

    python evaluation/bev_suite.py RUN [global options] <command ...>          (from the bev_fusion folder; RUN = runs/<name>)

    qual frame scene-0061/12 [--view overview debug outputs stages dashboard taps] [--history warmup|full|none]
    qual debug scene-0061/12                    overview + the stages of the chain (taps) of one frame
    qual random 6 [--seed 1]                    6 frames: a random video first, then a random frame in it
    qual videos 2 [--view dashboard]            2 random scenes as videos (the memory carried along)
    quant run [--n-scenes 20 | --spread 500]    numbers over many videos, stored per frame        quant table|per-scene|outliers|trace|report|timing ...
    list [scene]                                the val scenes and frames
    eval [--n 1000] [--warm] [--prior 500] [--images 8] [--time]   the evaluation of a run (replaces evaluate_run.py): runs the
                                                suite over the val frames, writes eval/metrics<tag>.json + eval/table<tag>.txt
                                                + eval/card<tag>.json (the metric card of decide.py / experiment.py, with its `provenance` block:
                                                GT version, evaluator code hash, frame hash, raw-data manifest hash, model; its `protocol` = GT signature,
                                                memory mode, control, weights: decide.py refuses cards that differ in a key both carry). A store started with other
                                                evaluator code (the Source's `evaluator_hash`) is not resumed: move it aside or use a new name
    report [--vs RUN ...] [--frames 6]          ONE PAGE for a person without code: state, resumes, the card's prediction / rule, curves, the numbers next to
                                                the --vs runs (first = control), cost, random frames + a video (made first when the run has none) and a
                                                review field per picture; -> RUN/report/index.html (autoresearch/tools/run_report.py; --ingest there
                                                takes the review.json the page exports)
    generalisation [--n N]                      the same evaluation on val and on the TRAIN split (N frames each; default = the largest n already
                                                evaluated on val, else 600; stored val numbers are reused) -> eval/table_train.txt + the page with the
                                                train-vs-val gap table; train = what the architecture can fit, the gap = what is lost on unseen scenes
    time                                        ms per frame of the chain (idle GPU only) -> eval/latency.json
    modules [--frames 3] [--threads 2]          module probe (evaluation/bev_modules.py; CPU unless --device is given): output statistics and dead channels of every
                                                module, the response when the camera / all radar detections / only the Doppler are removed, the velocity field on
                                                static and moving cells, a warm second frame for the memories, one PNG per frame -> eval/modules/
    timing [--n 30] [--force]                   ms per module at batch 1 + the forward / backward split of a training micro-batch (evaluation/bev_timing.py;
                                                idle GPU only, refused otherwise) -> eval/timing_modules.json
    memory [--clips 600]                        does the carried memory help? warm vs cold on the same frames -> eval/memory.json + txt
    compare RUN_A RUN_B [--tag _warm] [--n N] [--force]   paired scene bootstrap between two evaluated runs (b - a; runs evaluated
                                                before the suite: their scene_records pickles through evaluation/bev_legacy.py, only on
                                                the same frames); refused when the recorded GT signatures (evaluation: metrics / manifest,
                                                training: RUN/gt.json) differ, a warning when one is not recorded; --force overrides
    front REF SYSTEM [SYSTEM ...] [--x-max 50] [--n 1000] [--boot 1000] [--out FILE.md]   front-sector detection table of several evaluated
                                                runs (or LABEL=FILE.pkl records): AP at 0.5 / 1 / 2 / 4 m, mAP2, per range bin, difference to REF with the
                                                paired scene bootstrap; same GT / frame checks as compare (evaluation/front_compare.py)
Global options: --device cuda|cpu  --workers N (read-ahead processes, default 2)  --split val|train  --ckpt-dir DIR (moved checkpoints)
                --control shuffle_radar|zero_radar|shuffle_cam|zero_cam  --out DIR (default RUN/eval/suite)  --state carry|reset
Outputs: RUN/eval/suite/{qualitative, quant}; see autoresearch/evalsuite/EXTENSIONS.md chapter 5 for what each command writes.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, ".."))                       # so that `autoresearch` can be located from any cwd
sys.path.insert(0, os.path.join(ROOT, "..", "autoresearch"))
sys.path.insert(0, ROOT)
from evaluation import bev_adapter as A  # noqa: E402  (numpy only at import: the chain's code is imported later, after setup_paths)

HEADLINE_LAYERS = (("drivable_area", "iou@0"), ("drivable_area", "iou@0.5"), ("ped_crossing", "iou@0"), ("walkway", "iou@0"),
                   ("carpark_area", "iou@0"), ("lane_divider", "iou@0.5"), ("lane_divider", "iou@1"), ("road_divider", "iou@1"),
                   ("lane_divider:zc", "f1@0.5"), ("road_divider:zc", "f1@0.5"), ("lane_divider:zc", "field_mae"))


def tag_of(warm: bool, on_train: bool, control: str) -> str:
    return ("_warm" if warm else "") + ("_train" if on_train else "") + (f"_ctl_{control}" if control else "")


def _flat_name(m: str, rest: str, prefix_seg: str):
    if m == "seg":
        mask, r = rest.split("/", 1)
        return f"{prefix_seg}_{mask}/seg/{r}"
    if m in ("det", "detr", "curve", "detf", "detfp", "cov", "vel"):
        return f"{m}/{rest}"
    if m == "loss":
        return "val_loss/" + rest.split("/", 1)[1]
    return None


def to_legacy(table: dict, prefix_seg: str = "seg") -> dict:
    """suite table ('seg.fov/dt+0.00/drivable_area/all/all/iou@0', 'det.dt+0.00/vehicle/all/ap@2', ...) -> the flat names of the former
    evaluate_run metrics.json ('seg_fov/seg/dt+0.00/...', 'det/dt+0.00/...', 'detr/...', 'val_loss/total')"""
    flat = {}
    for k, v in table.items():
        f = _flat_name(*k.split(".", 1), prefix_seg)
        if f is not None:
            flat[f] = v
    return flat


def better_of(table: dict, metrics, prefix_seg: str = "seg") -> dict:
    """the card's "better" map of the flat names of to_legacy ({name: 'lower' | 'higher'}, card contract, read by
    autoresearch/tools/decide.py), from each metric's own direction(); counts (n, n_gt, n_tp_*) have no direction and are left out"""
    by = {m.name: m for m in metrics}
    out = {}
    for k in table:
        m, rest = k.split(".", 1)
        f, last = _flat_name(m, rest, prefix_seg), rest.rsplit("/", 1)[-1]
        if f is not None and m in by and last != "n" and not last.startswith("n_"):
            d = by[m].direction(rest)
            if d != "none":                                              # prevalence, thr_f1: not quality numbers
                out[f] = "lower" if d == "low" else "higher"
    return out


def ms_better(flat: dict) -> dict:
    """the timing numbers (ms/...) are lower-is-better; the idle flag and info/... have no direction"""
    return {k: "lower" for k, v in flat.items() if k.startswith("ms/") and isinstance(v, (int, float)) and not isinstance(v, bool)}


def prior_numbers(ptab: dict, metrics) -> tuple:
    """(flat, better) of the prior-map baseline: its map numbers only (prior_*); it predicts no boxes and no loss, and its det / detr / loss
    numbers carry the model's names, so they would overwrite the model's own"""
    flat, better = to_legacy(ptab, prefix_seg="prior"), better_of(ptab, metrics, prefix_seg="prior")
    return ({k: v for k, v in flat.items() if k.startswith("prior_")}, {k: v for k, v in better.items() if k.startswith("prior_")})


def contract_card(run: str, split: str, tag: str, flat: dict, better: dict, n_frames: int, gt_sig, state: str, control: str,
                  weights: str = "EMA") -> dict:
    """the metric card of this evaluation in the evaluator contract (autoresearch/schemas/card.schema.json): the file decide.py and
    experiment.py read (eval/card<tag>.json). The finite numbers of metrics.json (no info/*, no NaN, no flags) with their directions;
    commit = the run's code snapshot (code/SNAPSHOT.txt of run_ar.sh / run_cfg.sh, '+?' when unknown); seed from the run's config.json;
    protocol = what the numbers were measured under: the GT signature scored against, the memory mode (state: reset = cold, carry = warm), the
    sanity control and the weights (decide.py compares the protocol key by key and refuses a comparison across protocols: a warm card is not
    comparable to a cold one with the same frames; a card from before a key existed does not carry it and is not refused for that). The prior-map
    baseline's frames are not part of it: its numbers are extra keys (prior_*)"""
    metrics = {k: float(v) for k, v in flat.items()
               if not k.startswith("info/") and isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)}
    try:
        words = open(os.path.join(run, "code", "SNAPSHOT.txt")).read().split()
        commit = words[words.index("commit") + 1]
    except (OSError, ValueError, IndexError):
        commit = "+?"
    card = {"schema": 1, "name": os.path.basename(os.path.normpath(run)) + tag, "split": split, "frames": int(n_frames), "metrics": metrics,
            "better": {k: v for k, v in better.items() if k in metrics}, "commit": commit, "date": time.strftime("%Y-%m-%d %H:%M"),
            "protocol": {"gt_signature": gt_sig, "state": state, "control": control or None, "weights": weights}}
    seed = (_read_json(os.path.join(run, "config.json")) or {}).get("seed")
    if isinstance(seed, (int, str)) and not isinstance(seed, bool):
        card["seed"] = seed
    return card


def provenance_start(suite) -> dict:
    """what is fixed when the evaluation STARTS: the hash of the evaluator's source files and of the checkpoint. The block is written 15 - 30 minutes later, in a
    working tree that other chats edit; a change in between is then marked in the block, not hidden. A collector that cannot be loaded is said here, at minute 0."""
    try:
        mod = A._provenance_module()
        ckpt = getattr(suite.subject, "ckpt", None)
        return {"tree": A.evaluator_tree_hash(mod),
                "ckpt_path": ckpt, "ckpt": mod.file_hash(ckpt) if ckpt and os.path.isfile(ckpt) else None}
    except Exception as ex:
        print(f"WARNING: provenance collector not usable ({type(ex).__name__}: {ex}); the card will carry a gap", flush=True)
        return {"error": f"{type(ex).__name__}: {ex}"}


def data_use(split: str, store, avail=None) -> dict:
    """which data an evaluation used, readable without the code: the split is the nuScenes norm (nuscenes.utils.splits.create_splits_scenes: train 700 /
    val 150 scenes); only the scenes whose features are cached can be run, so `available` is smaller than `official`; `scenes` are the ones the frames of
    this evaluation came from (their names, so another person can look them up); the test split of nuScenes has no public ground truth and is not used"""
    try:
        from nuscenes.utils.splits import create_splits_scenes
        official = set(create_splits_scenes().get(split, []))
    except Exception:                                    # no devkit: the counts stay unknown, the names stay
        official = set()
    used = sorted(store.scene_names())
    return {"split_rule": f"nuScenes official split '{split}' (nuscenes.utils.splits.create_splits_scenes), restricted to the scenes with cached features; "
                          "the nuScenes test split has no public ground truth and is not used",
            "scenes_official": len(official) or None, "scenes_available": len(avail) if avail is not None else None, "scenes_evaluated": len(used),
            "frames_evaluated": len(store.frames), "scenes_not_in_official_split": [s for s in used if official and s not in official],
            "scene_names": used}


def provenance_block(a, e, store, subj: dict, gt_sig, tag: str, state: str, start: dict = None, avail=None) -> dict:
    """the `provenance` block of the card (design autoresearch/reproducibility.md, collector autoresearch/tools/provenance.py): GT version and
    signature, the hash of the evaluator's code (evaluation/ and autoresearch/evalsuite/) with the files that differ from its commit, the frame
    list, the raw-data manifest, the model (run, checkpoint hash and step, config hash, its training GT), the settings and the environment of THIS
    interpreter. `start` = provenance_start(): when the evaluator's source hash or the checkpoint's hash at the end differs from the one at the start,
    the block says so (evaluator.dirty = 'unknown', changed_during_run) and decide.py refuses the card as unchecked. A failure to collect never stops an
    evaluation: the block then holds the error, a WARNING is printed, and decide.py refuses the card as unchecked."""
    try:
        mod = A._provenance_module()
        start = start or {}
        cfg, ckpt = os.path.join(a.run, "config.json"), start.get("ckpt_path") or subj.get("checkpoint")
        if not (ckpt and os.path.isfile(ckpt)) and subj.get("checkpoint"):    # a checkpoint that was moved (--ckpt-dir) is found there
            for d in (getattr(a, "ckpt_dir", None), a.run):
                if d and os.path.isfile(os.path.join(d, subj["checkpoint"])):
                    ckpt = os.path.join(d, subj["checkpoint"])
                    break
        man = getattr(store, "manifest", None) or {}
        pv = mod.collect(gt_key=gt_sig, split=a.split, tokens=[fr["key"] for fr in store.frames], evaluator=list(A.EVALUATOR_PATHS),
                         run=os.path.basename(os.path.normpath(a.run)), ckpt=ckpt, config=cfg, step=subj.get("step"), run_dir=a.run,
                         seed=(_read_json(cfg) or {}).get("seed"),
                         settings={"tag": tag, "n": e.n, "warm": bool(e.warm), "prior": e.prior, "control": a.control or None, "state": state,
                                   "weights": "EMA", "device": subj.get("device") or getattr(a, "device", None), "code": subj.get("code"),
                                   "store": {k: man.get(k) for k in ("created", "resumed", "commit")}})
        pv.setdefault("data", {})["use"] = data_use(a.split, store, avail)
        ver = (_read_json(cfg) or {}).get("version")                          # the collector leaves the raw-data version "unknown" without a GT store that names it
        if ver and pv["data"].get("dataset") in (None, "unknown"):
            pv["data"]["dataset"] = ver
        ev, sub = pv.get("evaluator", {}), pv.get("subject", {})
        if start.get("tree") and ev.get("code_hash") != start["tree"]:       # the evaluator's code changed while the evaluation ran
            ev.update(dirty="unknown", changed_during_run=True, code_hash_at_start=start["tree"])
        if start.get("ckpt") and sub.get("ckpt_hash") is None:               # the checkpoint is gone at the end (pruned, archived): the hash of the start stands
            sub.update(ckpt=os.path.basename(start["ckpt_path"]), ckpt_hash=start["ckpt"], gone_at_end=True)
        elif start.get("ckpt") and sub.get("ckpt_hash") != start["ckpt"]:
            sub.update(ckpt_hash="unknown", changed_during_run=True)
        return pv
    except Exception as ex:                                                  # a marked gap in the record, not a lost evaluation
        err = f"{type(ex).__name__}: {ex}"
        print(f"WARNING: provenance not collected: {err}", flush=True)
        return {"schema": 1, "error": err}


def headline_table(flat: dict) -> str:
    keys = []
    for pre in ("seg_fov", "prior_fov"):
        for dt in ("dt+0.00", "dt+0.50", "dt-0.50"):
            for layer, metric in HEADLINE_LAYERS:
                keys.append(f"{pre}/seg/{dt}/{layer}/all/all/{metric}")
    for pre in ("seg_obj", "prior_obj"):
        for dt in ("dt+0.00", "dt+0.50"):
            for layer in ("vehicle", "pedestrian", "two_wheeler"):
                keys.append(f"{pre}/seg/{dt}/{layer}/all/all/iou@0")
    for dp in ("det", "detr"):                                      # detr = boxes refined by S5c (only for models with a refiner)
        for dt in ("dt+0.00", "dt+0.50", "dt-0.50"):
            keys.append(f"{dp}/{dt}/mean/all/mAP")
            for cls in ("vehicle", "pedestrian", "two_wheeler"):
                for m in ("ap@2", "ap@4", "ate", "ase", "aoe", "ave", "n_gt", "ave_moving", "ave0_moving", "ave_skill_moving", "ave_slow_moving",
                          "n_tp_moving", "ave_static", "ave0_static", "ave_skill_static", "n_tp_static"):
                    keys.append(f"{dp}/{dt}/{cls}/all/{m}")
    for layer in ("drivable_area", "walkway", "carpark_area", "ped_crossing", "lane_divider", "road_divider", "vehicle", "pedestrian"):      # --curves only
        keys += [f"curve/dt+0.00/{layer}/{m}" for m in ("ap", "roc_auc", "ece")]
    keys += [f"curve/dt+0.00/velocity/{sub}/{m}" for sub in ("obj", "moving") for m in ("epe", "rmse", "r2", "bias")]
    for dp in ("detf", "detfp"):                                    # --fusion only: the fusion-area rule (D37 / D38), a separate label from `det`
        keys += [f"{dp}/dt+0.00/mean/all/mAP"] + [f"{dp}/dt+0.00/{cls}/all/{m}" for cls in ("vehicle", "pedestrian", "two_wheeler") for m in ("ap@2", "ap@4", "ate", "n_gt")]
    keys += [f"cov/dt+0.00/{cls}/{c}/{m}" for cls in ("vehicle", "pedestrian") for c in ("all", "care", "partial", "dont_care", "camera", "cam_radar", "cam_only", "radar_only", "neither")
             for m in ("n_gt", "recall2m")]
    keys += [f"vel/dt+0.00/centre/{s}/moving/{m}" for s in ("field", "objhead", "zero") for m in ("epe", "n")]     # --velocity only
    keys += [f"vel/dt+0.00/cells/{c}/epe" for c in ("static", "vehicle", "vehicle_moving", "pedestrian_moving")]
    keys += [k for k in flat if k.startswith(("val_loss/", "ms/", "info/"))]
    lines = []
    for k in keys:
        if k in flat:
            v = flat[k]
            lines.append(f"{k:60s} {v:10.4f}" if isinstance(v, float) else f"{k:60s} {v!s:>10}")
    return "\n".join(lines)


def _read_json(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def pickle_meta(path: str):
    """the meta dict of a legacy scene_records pickle (frames, gt_signature), None if there is none or it cannot be read"""
    import pickle
    try:
        with open(path, "rb") as f:
            return pickle.load(f).get("meta")
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ValueError):
        return None


def stored_gt(stored: str):
    """the GT signature a stored suite run was scored on: None = no stored run, 'unrecorded' = made before the signature was kept"""
    man = _read_json(os.path.join(stored, "manifest.json"))
    return None if man is None else ((man.get("source") or {}).get("gt_signature") or "unrecorded")


def gt_conflict(stored: str, gt_sig, force_unrecorded: bool):
    """(refusal message or None, the label of the result): a resume computes only the missing frames, so a stored run of another GT
    would carry the current label. A run of another RECORDED GT is always refused (the evalsuite store refuses it too); a run made
    before the signature was recorded is resumed only with force_unrecorded, labelled 'mixed:unrecorded+<current>'."""
    old = stored_gt(stored)
    if old is None or old == gt_sig:
        return None, gt_sig
    if old == "unrecorded" and force_unrecorded:
        return None, f"mixed:unrecorded+{gt_sig}"
    return (f"refused: the stored run {stored} was scored on GT {old}, the current GT is {gt_sig}; move it aside for a fresh evaluation"
            + (" (or --force-resume: it was made before the signature was recorded)" if old == "unrecorded" else "")), gt_sig


def train_gt(run: str):
    """the GT signature a run trained on (runs/<name>/gt.json, written by training/train.py since 2026-10-31); None = not recorded"""
    return (_read_json(os.path.join(run, "gt.json")) or {}).get("gt_signature")


def evaluator_conflict(stored: str, hash_now):
    """refusal text or None: a resume computes only the missing frames, so a store started with other evaluator code would be finished with this code and
    carry one hash on a mixture of both. A store made before the Source carried `evaluator_hash` has no value to compare: the evalsuite store tolerates the new
    key with its warning (resume of ... description gained keys)"""
    old = (_read_json(os.path.join(stored, "manifest.json")) or {}).get("source") or {}
    if "evaluator_hash" not in old or old["evaluator_hash"] == hash_now:
        return None
    return (f"refused: evaluator code changed since this store was started ({stored}: {old['evaluator_hash']} -> {hash_now}): use a new --name "
            "(`eval` / `memory` name their stores themselves: move that directory aside, or `eval --force-resume` to finish it with this code)")


def quant_run(suite, force_evaluator: bool = False, **kw):
    """suite.quant.run with the evaluator check of evaluator_conflict (SystemExit with the text, before any frame is run); force_evaluator: the person accepts
    a store finished with other evaluator code (the evaluator hash is the only identity key that may differ)"""
    refusal = None if force_evaluator else evaluator_conflict(os.path.join(suite.quant.dir, kw["name"]), suite.source.describe().get("evaluator_hash"))
    if refusal:
        raise SystemExit(refusal)
    return suite.quant.run(ignore={"source": ("evaluator_hash",)} if force_evaluator else None, **kw)


def make_suite(a, prior: int = 0, state: str = None, evaluator: str = None):
    from evalsuite import EvalSuite
    from evaluation.bev_adapter import build
    source, subject, task = build(a.run, device=a.device, split=a.split, workers=a.workers, ckpt_dir=a.ckpt_dir, control=a.control, prior=prior, evaluator=evaluator)
    out = a.out or os.path.join(a.run, "eval", "suite")
    task.curves = bool(getattr(a, "curves", False))                          # --curves: PR / ROC / calibration / velocity regression next to the standard numbers
    task.fusion = bool(getattr(a, "fusion", False))                          # --fusion: boxes scored on the fusion area (detf / detfp) and the coverage split (cov), D37 / D38
    task.velocity = bool(getattr(a, "velocity", False))                      # --velocity: field per cell set and box-centre velocity (vel, bev_velocity.py)
    return EvalSuite(source, subject, task, out=out, state=state or a.state)


def cmd_eval(a, rest) -> int:
    p = argparse.ArgumentParser(prog="eval")
    p.add_argument("--n", type=int, default=0, help="val frames spread over all scenes (0 = all)")
    p.add_argument("--warm", action="store_true", help="the memory carried along whole scenes (needs --n 0 or --scenes)")
    p.add_argument("--prior", type=int, default=0, help="train frames of the prior-map baseline (0 = off)")
    p.add_argument("--images", type=int, default=0)
    p.add_argument("--time", action="store_true")
    p.add_argument("--scenes", nargs="*", default=None)
    p.add_argument("--force-resume", action="store_true",
                   help="resume a stored run made before the GT signature was recorded (labelled 'mixed'), or finish a store that was started with other evaluator code; "
                        "another recorded GT is always refused")
    e = p.parse_args(rest)
    tag = tag_of(e.warm, a.split == "train", a.control)
    state = "carry" if e.warm else "reset"
    suite = make_suite(a, state=state)
    prov0 = provenance_start(suite)                                         # the evaluator's and the checkpoint's hash NOW, before any frame is scored
    from evalsuite.sampling import Sampler
    frames = Sampler(suite.source, 0).spread(e.n, e.scenes) if e.n else None
    name = f"eval{tag}{'_curves' if getattr(a, 'curves', False) else ''}{'_fusion' if getattr(a, 'fusion', False) else ''}{'_vel' if getattr(a, 'velocity', False) else ''}_n{e.n or 'all'}"      # curve / fusion numbers need their own store: an eval made without them is not resumed
    desc = suite.source.describe()
    gt_now = desc.get("gt_signature")                                    # the GT scored against (autogt is read live, not pinned)
    quant_dir = os.path.join(a.out or os.path.join(a.run, "eval", "suite"), "quant")
    stores = [os.path.join(quant_dir, name)] + ([os.path.join(quant_dir, f"prior{e.prior}_n{e.n or 'all'}")] if e.prior else [])    # the prior baseline's stored run as well
    refusal, gt_sig = gt_conflict(stores[0], gt_now, e.force_resume)
    for st_dir in stores[1:]:
        refusal = refusal or gt_conflict(st_dir, gt_now, e.force_resume)[0]
    for st_dir in stores:                                                  # before any frame is run, not after the first store has finished
        refusal = refusal or (None if e.force_resume else evaluator_conflict(st_dir, desc.get("evaluator_hash")))
    if refusal:
        print(refusal, flush=True)
        suite.close()
        return 2
    t0 = time.time()
    from evaluation.bev_views import gt_boxes
    groups = list(suite.task.groups)
    det_rec, detr_rec, seen = {}, {}, set()

    def keep_boxes(sample, pred):                          # the boxes of every frame, for the legacy scene_records.pkl
        seen.add(sample.ref.key)
        sc = sample.ref.scene
        for dt, o in pred.outputs["dt"].items():
            gb = gt_boxes(sample.reference[dt], groups)
            det_rec.setdefault(sc, {}).setdefault(dt, []).append((list(o["dets"]), gb))
            if o.get("detr") is not None:
                detr_rec.setdefault(sc, {}).setdefault(dt, []).append((list(o["detr"]), gb))
    store = quant_run(suite, force_evaluator=e.force_resume, scenes=e.scenes if not e.n else None, frames=frames, state=state, name=name, progress=50, on_frame=keep_boxes)
    tab = suite.quant.table(store)
    flat, better = to_legacy(tab), better_of(tab, suite.task.metrics())
    subj = suite.subject.describe()
    flat["info/step"], flat["info/n_samples"], flat["info/n_scenes"] = subj.get("step"), len(store.frames), len(store.scene_names())
    lat = None
    if e.time:
        lat = _latency(suite, a)
        flat.update({f"ms/{k}": v for k, v in lat.items() if isinstance(v, (int, float)) and k != "n"})
    if e.prior:
        psuite = make_suite(a, prior=e.prior, state="reset", evaluator=desc.get("evaluator_hash"))       # the main suite's hash: both stores were checked against it above
        pstore = quant_run(psuite, force_evaluator=e.force_resume, frames=frames or [f for s in suite.source.scenes() for f in suite.source.frames(s)], state="reset",
                                  name=f"prior{e.prior}_n{e.n or 'all'}", progress=50)
        pflat, pbetter = prior_numbers(psuite.quant.table(pstore), psuite.task.metrics())
        flat.update(pflat)
        better.update(pbetter)
        psuite.close()
    better.update(ms_better(flat))
    if e.images:
        for ref in Sampler(suite.source, 0).spread(e.images):
            suite.qual.frame(ref, ("overview", "outputs"), out_dir=os.path.join(a.run, "eval", "images"),
                             history="full" if e.warm else "none", state=state)
    out_dir = os.path.join(a.run, "eval")
    os.makedirs(out_dir, exist_ok=True)
    n_tag = f"_n{e.n or 'all'}"
    os.makedirs(os.path.join(out_dir, "by_n"), exist_ok=True)

    def write_both(stem, ext, text):     # by_n/<stem><tag>_n<n> keeps the eval of each n (an early check with --n 200 no longer replaces
        for path in (os.path.join(out_dir, "by_n", f"{stem}{tag}{n_tag}.{ext}"),   # the 1000-frame numbers); the plain name is the
                     os.path.join(out_dir, f"{stem}{tag}.{ext}")):                  # latest eval, as before (by_n/: globs such as
            with open(path, "w") as f:                                              # metrics_ctl_*.json do not see the copies)
                f.write(text)
    write_both("metrics", "json", json.dumps({"run": a.run, "checkpoint": subj.get("checkpoint"), "split": a.split, "weights": "EMA",
                                              "memory": "carried over the whole scene" if e.warm else "empty (cold)", "code": subj.get("code"),
                                              "gt_signature": gt_sig, "train_gt_signature": train_gt(a.run),
                                              "suite_run": os.path.relpath(store.path, a.run), "metrics": flat, "better": better,
                                              "latency": lat}, indent=1, default=float))
    card = contract_card(a.run, a.split, tag, flat, better, len(store.frames), gt_sig, state, a.control)         # for decide.py
    card["provenance"] = provenance_block(a, e, store, subj, gt_sig, tag, state, start=prov0, avail=list(suite.source.scenes()))  # which GT, evaluator, frames, data and model made it
    write_both("card", "json", json.dumps(card, indent=1))
    from evaluation import bev_legacy                                    # the n is in the name: an eval of another n never overwrites it
    rec_path, plain = os.path.join(out_dir, f"scene_records{tag}{n_tag}.pkl"), os.path.join(out_dir, f"scene_records{tag}.pkl")
    keys = [fr["key"] for fr in store.frames]
    import shutil                                                        # the plain name too, always replaced: the temp readers (collect.py,
    if seen >= set(keys):                                                # build_tables.py, ...) read it next to metrics<tag>.json
        bev_legacy.write_scene_records(rec_path, suite.task.metrics()[0], store, det_rec, detr_rec, meta={"frames": keys, "gt_signature": gt_sig})
        shutil.copyfile(rec_path, plain)
    elif (pickle_meta(rec_path) or {}).get("frames") == keys and pickle_meta(rec_path).get("gt_signature") == gt_sig:
        shutil.copyfile(rec_path, plain)                                 # a re-run of a finished store (e.g. --time): its pickle still fits
    else:                                # a resumed store: the frames stored before were not run in this process, so the boxes of their
        print(f"WARNING: scene_records not written: this process saw {len(seen & set(keys))} of {len(keys)} frames (resumed store), its "
              f"detection part would cover a subset of the scenes; evaluate into a fresh store for the legacy pickle", flush=True)
        for stale in (rec_path, plain):
            if os.path.exists(stale):
                os.replace(stale, stale + ".stale")
    table = headline_table(flat)
    write_both("table", "txt", table + "\n")
    print(table)
    print(f"done: {len(store.frames)} {a.split} frames, {len(store.scene_names())} scenes, {time.time() - t0:.0f} s -> {out_dir} (suite run {store.path})")
    suite.close()
    return 0


def _latency(suite, a) -> dict:
    """ms per frame of the cached chain at batch 1 (idle GPU only: the load before and after is recorded, 'idle' says < 10 %)"""
    import torch
    lat = suite.quant.timing()
    lat["idle"] = bool(lat.get("load", {}).get("gpu_util_percent", 100) < 10)
    lat["gpu"] = torch.cuda.get_device_name(0) if suite.subject.device.startswith("cuda") else "cpu"
    return lat


def cmd_time(a, rest) -> int:
    suite = make_suite(a, state="reset")
    lat = _latency(suite, a)
    os.makedirs(os.path.join(a.run, "eval"), exist_ok=True)
    with open(os.path.join(a.run, "eval", "latency.json"), "w") as f:
        json.dump({"run": a.run, "checkpoint": suite.subject.describe().get("checkpoint"), "gpu": lat["gpu"], "latency": lat}, f, indent=1,
                  default=float)
    print(json.dumps(lat, indent=1, default=float))
    suite.close()
    return 0


MEMORY_ROWS = (("drivable IoU@0 (in view)", "seg_fov/seg/{dt}/drivable_area/all/all/iou@0"),
               ("drivable IoU@0 (outside view)", "seg_out/seg/{dt}/drivable_area/all/all/iou@0"),
               ("walkway IoU@0 (outside view)", "seg_out/seg/{dt}/walkway/all/all/iou@0"),
               ("lane-div mask IoU@1 (outside view)", "seg_out/seg/{dt}/lane_divider/all/all/iou@1"),
               ("drivable IoU@0 (camera or radar)", "seg_seen/seg/{dt}/drivable_area/all/all/iou@0"),
               ("walkway IoU@0", "seg_fov/seg/{dt}/walkway/all/all/iou@0"),
               ("lane-divider mask IoU@1", "seg_fov/seg/{dt}/lane_divider/all/all/iou@1"),
               ("vehicle coverage IoU", "seg_obj/seg/{dt}/vehicle/all/all/iou@0"),
               ("vehicle AP@2", "det/{dt}/vehicle/all/ap@2"), ("pedestrian AP@2", "det/{dt}/pedestrian/all/ap@2"),
               ("vehicle ATE", "det/{dt}/vehicle/all/ate"), ("vehicle AVE", "det/{dt}/vehicle/all/ave"))


def interval_text(v: dict) -> str:
    """the interval of one paired-bootstrap row (evalsuite stats): '95% [lo, hi]', '(excludes 0)', and the draws it rests on (n_valid / n_draws). A resample with
    nothing to score is dropped, so with n_valid < n_draws the interval is conditional on the scorable resamples: the row says so"""
    s = f"95% [{v['lo']:+.4f}, {v['hi']:+.4f}]" + ("  (excludes 0)" if v["lo"] > 0 or v["hi"] < 0 else "") + f"  draws {v['n_valid']}/{v['n_draws']}"
    return s + ("  WARNING: only some resamples were scorable, the interval is conditional on them" if v["n_valid"] < v["n_draws"] else "")


def cmd_memory(a, rest) -> int:
    """Does the carried world memory help? The same val frames twice: 'cold' = every frame from an empty memory, 'warm' = from an
    empty memory with the previous seq_len - 1 keyframes run first (what the clip-trained network saw); numbers per dt, and the
    paired scene bootstrap of the headline numbers (warm - cold). -> RUN/eval/memory.json + memory.txt"""
    p = argparse.ArgumentParser(prog="memory")
    p.add_argument("--clips", type=int, default=600, help="val frames (with a full history) spread over all scenes")
    m = p.parse_args(rest)
    suite = make_suite(a, state="reset")
    L = int(suite.subject.cfg.seq_len)
    if L < 2:
        raise SystemExit(f"{a.run}: seq_len = {L}, nothing carried")
    from evalsuite.sampling import Sampler
    frames = Sampler(suite.source, 0).spread(m.clips, min_index=L - 1)
    cold = quant_run(suite, frames=frames, state="reset", name=f"memory_cold_n{m.clips}", progress=50)
    warm = quant_run(suite, frames=frames, state="warmup", name=f"memory_warm_n{m.clips}", progress=50)
    flat = {f"{w}/{k}": v for w, st in (("warm", warm), ("cold", cold)) for k, v in to_legacy(suite.quant.table(st)).items()}
    use_ref = any(k.startswith("warm/detr/") for k in flat)
    lines = [f"{a.run} step {suite.subject.describe().get('step')}: warm vs cold memory on {len(frames)} val frames with {L - 1} keyframes of "
             f"history (clips of {L} keyframes; EMA, VAL)"]
    for dt in ("dt+0.00", "dt+0.50", "dt-0.50"):
        rows = list(MEMORY_ROWS) + ([("vehicle AP@2 refined", "detr/{dt}/vehicle/all/ap@2"), ("vehicle ATE refined", "detr/{dt}/vehicle/all/ate")]
                                    if use_ref else [])
        lines.append(dt)
        for label, key in rows:
            kw, kc = f"warm/{key.format(dt=dt)}", f"cold/{key.format(dt=dt)}"
            if kw in flat and kc in flat:
                lines.append(f"  {label:36s} warm {flat[kw]:8.4f}  cold {flat[kc]:8.4f}  warm - cold {flat[kw] - flat[kc]:+8.4f}")
    cmp = suite.quant.compare(cold, warm, boot=1000)
    lines.append("paired scene bootstrap of the headline numbers, warm - cold (dt 0):")
    for k, v in cmp["metrics"].items():
        lines.append(f"  {k:62s} cold {v['a']:.4f}  warm {v['b']:.4f}  diff {v['diff']:+.4f}  {interval_text(v)}")
    txt = "\n".join(lines)
    out = os.path.join(a.run, "eval")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "memory.json"), "w") as f:
        json.dump({"run": a.run, "checkpoint": suite.subject.describe().get("checkpoint"), "n_clips": len(frames), "seq_len": L,
                   "metrics": flat, "paired": cmp["metrics"]}, f, indent=1, default=float)
    with open(os.path.join(out, "memory.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)
    suite.close()
    return 0


def _task_only(run: str, curves: bool = False, fusion: bool = False, velocity: bool = False):
    """the Task of a run without loading its checkpoint (metrics definitions only; curves / fusion / velocity: the metric families of a store made with --curves / --fusion / --velocity)"""
    from evaluation import bev_adapter as B
    from evaluation.bev_task import BEVTask
    os.chdir(ROOT)
    B.setup_paths(run)
    ns = B._chain()
    with open(os.path.join(run, "config.json")) as f:
        cfg = ns.T.TrainConfig(**json.load(f))
    model, _, _ = ns.T.build(cfg)
    refine = bool(cfg.model.get("refine")) and hasattr(model, "boxes")
    task = BEVTask(model.grid, model.spec, [g for g, _ in ns.NUSC_GROUPS], refine=refine)
    task.curves, task.fusion, task.velocity = curves, fusion, velocity
    return task


def gt_check(run_a: str, run_b: str, ids: dict, force: bool = False) -> list:
    """refuse (SystemExit) when the recorded GT identities of two runs differ; ids = {"evaluation GT": (sig_a, sig_b), "training GT":
    (sig_a, sig_b)}, None = not recorded (runs trained / evaluated before 2026-10-31: a warning only). force: warn instead of refusing.
    -> the warning lines"""
    warn, bad = [], []
    for what, (sa, sb) in ids.items():
        if sa is None or sb is None:
            who = " and ".join(r for r, s in ((run_a, sa), (run_b, sb)) if s is None)
            warn.append(f"WARNING: {what} signature not recorded for {who}: the GT identity is unchecked")
        elif sa != sb:
            bad.append(f"{what} {sa} ({run_a}) vs {sb} ({run_b})")
    if bad:
        msg = "the two runs used different GT: " + "; ".join(bad)
        if not force:
            raise SystemExit(f"compare refused: {msg}. Their numbers are not comparable: evaluate (train) both on the same autogt "
                             "version, or pass --force")
        warn.append(f"WARNING (--force): {msg}")
    return warn


def legacy_record(run: str, tag: str, n: str):
    """the pickle record of a run for the legacy compare -> (file tag, frame keys or None, number of frames or None, evaluation GT
    signature or None), or None. New evaluations write scene_records<tag>_n<N>.pkl with the frame keys inside; an older
    scene_records<tag>.pkl was written together with metrics<tag>.json (its suite store names the frames; before the suite: unknown)."""
    import pickle
    path = os.path.join(run, "eval", f"scene_records{tag}_n{n}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            meta = pickle.load(f).get("meta") or {}
        fr = meta.get("frames")
        return f"{tag}_n{n}", fr, (len(fr) if fr is not None else None), meta.get("gt_signature")
    if not os.path.exists(os.path.join(run, "eval", f"scene_records{tag}.pkl")):
        return None
    m = _read_json(os.path.join(run, "eval", f"metrics{tag}.json")) or {}
    frames, fj = None, os.path.join(run, m["suite_run"], "frames.jsonl") if m.get("suite_run") else ""
    if fj and os.path.exists(fj):
        with open(fj) as f:
            frames = [json.loads(line)["key"] for line in f if line.strip()]
    n_rec = len(frames) if frames is not None else (m.get("metrics") or {}).get("info/n_samples")
    return tag, frames, n_rec, m.get("gt_signature")


def frames_check(run_a: str, run_b: str, ra, rb, n: str, force: bool = False) -> list:
    """the legacy records hold per-scene sums, so they can only be paired when both runs saw the SAME frames: refuse (SystemExit)
    when the frame sets differ, when a record's frame count is not the requested --n, or when only one side names its frames;
    two records without frame keys (both evaluated before the suite) need equal frame counts. force: warn instead. -> warnings"""
    (_, fa, na, _), (_, fb, nb, _) = ra, rb
    bad, warn = [], []
    for run, nr in ((run_a, na), (run_b, nb)):
        if n != "all" and nr is not None and int(nr) != int(n):
            bad.append(f"{run} holds {nr} frames, not --n {n}")
    if fa is not None and fb is not None:
        if set(fa) != set(fb):
            bad.append(f"different frame sets ({len(set(fa) & set(fb))} common of {len(set(fa))} / {len(set(fb))})")
    elif fa is not None or fb is not None:
        bad.append(f"the frames of {run_a if fa is None else run_b} are not recorded (evaluated before the suite): their identity "
                   "cannot be checked; evaluate it with `bev_suite.py RUN eval --n N`")
    elif na is None or nb is None or int(na) != int(nb):
        bad.append(f"frame counts {na} / {nb} (records without frame keys)")
    else:
        warn.append(f"note: neither record names its frames (both evaluated before the suite); same count {na} assumed to be the same frames")
    if bad:
        msg = "; ".join(bad)
        if not force:
            raise SystemExit(f"compare refused (legacy records): {msg}. Or pass --force")
        warn.append(f"WARNING (--force): {msg}")
    return warn


COMPARE_IGNORE = {"source": ("feature_cache", "control", "evaluator_hash")}   # the arms may differ in what the model is fed (t8: features); the evaluator hash only guards a resume


def cmd_compare(argv) -> int:
    p = argparse.ArgumentParser(prog="compare")
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("--tag", default="")
    p.add_argument("--n", default="all", help="the n of the evaluations (as in the store name evalTAG_nN)")
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--det-a", default="det", help="legacy records only: box set of run a (det | detr)")
    p.add_argument("--det-b", default="det")
    p.add_argument("--dt", type=float, default=0.0, help="legacy records only: target time of the detection numbers")
    p.add_argument("--force", action="store_true", help="compare even when the GT signatures or (legacy records) the frame sets differ")
    p.add_argument("--curves", action="store_true", help="compare the stores made with `eval --curves` (PR / ROC / calibration / velocity numbers are bootstrapped too)")
    p.add_argument("--fusion", action="store_true", help="compare the stores made with `eval --fusion` (detf / detfp / cov)")
    p.add_argument("--velocity", action="store_true", help="compare the stores made with `eval --velocity` (vel)")
    c = p.parse_args(argv)
    from evalsuite.quantitative import Quantitative
    name = f"eval{c.tag}{'_curves' if c.curves else ''}{'_fusion' if c.fusion else ''}{'_vel' if c.velocity else ''}_n{c.n}"
    stores = [os.path.join(r, "eval", "suite", "quant", name, "manifest.json") for r in (c.run_a, c.run_b)]
    train_ids = (train_gt(c.run_a), train_gt(c.run_b))
    if not all(os.path.exists(s) for s in stores):
        ra, rb = (legacy_record(r, c.tag, c.n) for r in (c.run_a, c.run_b))
        if ra is not None and rb is not None:                   # runs evaluated before the suite: the former pickle records
            if ra[0] != rb[0]:
                raise SystemExit(f"compare refused (legacy records): {c.run_a} has scene_records{ra[0]}.pkl, {c.run_b} scene_records{rb[0]}.pkl"
                                 " - one of them was evaluated before the suite; evaluate it with `bev_suite.py RUN eval --n N`")
            warn = gt_check(c.run_a, c.run_b, {"evaluation GT": (ra[3], rb[3]), "training GT": train_ids}, c.force)
            warn += frames_check(c.run_a, c.run_b, ra, rb, c.n, c.force)
            for w in warn:
                print(w)
            from evaluation import bev_legacy
            bev_legacy.compare(c.run_a, c.run_b, c.boot, c.seed, c.det_a, c.det_b, c.dt, ra[0])
            return 0
        raise SystemExit(f"no evaluation '{name}' of both runs ({stores}); run `bev_suite.py RUN eval --n ...` on both")
    mans = [_read_json(s) or {} for s in stores]
    for w in gt_check(c.run_a, c.run_b, {"evaluation GT": tuple((m.get("source") or {}).get("gt_signature") for m in mans),
                                         "training GT": train_ids}, c.force):
        print(w)
    hashes = [(m.get("source") or {}).get("evaluator_hash") for m in mans]
    if hashes[0] != hashes[1]:                                     # not a refusal (the stores were made at different times: decide.py's provenance check judges the cards)
        print(f"WARNING: the two evaluations were made with different evaluator code ({c.run_a}: {hashes[0]}, {c.run_b}: {hashes[1]}); "
              "part of a difference may come from the evaluator, not from the runs")
    task = _task_only(c.run_a, c.curves, c.fusion, c.velocity)
    q = Quantitative(SimpleNamespace(out=os.path.join(c.run_a, "eval", "suite"), task=task))
    res = q.compare(os.path.join(c.run_a, "eval", "suite", "quant", name), os.path.join(c.run_b, "eval", "suite", "quant", name), c.boot, c.seed,
                    force=c.force, ignore=COMPARE_IGNORE)
    print(f"paired scene bootstrap ({c.boot} resamples), b - a; a = {c.run_a}, b = {c.run_b}; val; {res['n_common_frames']} common frames")
    for k, v in res["metrics"].items():
        print(f"{k:62s} a {v['a']:.4f}  b {v['b']:.4f}  b-a {v['diff']:+.4f}  {interval_text(v)}")
    return 0


REPORT_NOTE = ("maps are scored on `seg_fov` = the grid cells the front camera sees (62 % of the grid; today's fusion area, D37), objects inside "
               "+-32.3 deg and x < 60 m; `det`: boxes without a lidar / radar point are dropped from the GT, a detection on one still counts as a false "
               "detection (`detf` / `detfp` of `eval --fusion` make that two-sided, D38). dt +-0.5 s = the neighbouring keyframe in its own ego frame.")


def cmd_report(a, rest) -> int:
    """pictures (6 random frames + one video) when the run has none yet, then the one-page report (autoresearch/tools/run_report.py)"""
    import subprocess
    p = argparse.ArgumentParser(prog="report", allow_abbrev=False)
    p.add_argument("--vs", nargs="*", default=[], help="reference runs, the first one is the control of the delta column")
    p.add_argument("--frames", type=int, default=6)
    p.add_argument("--no-pictures", action="store_true")
    p.add_argument("--val-table", default="")
    p.add_argument("--train-table", default="")
    r = p.parse_args(rest)
    suite_dir = os.path.join(a.run, "eval", "suite")
    have = [f for f in glob.glob(os.path.join(suite_dir, "**", "*.png"), recursive=True) if os.sep + "quant" + os.sep not in f]
    if not r.no_pictures and not have:
        g = ["--device", a.device, "--workers", str(a.workers), "--split", a.split, "--state", a.state]
        for cmd in (["qual", "random", str(r.frames), "--seed", "1"], ["qual", "videos", "1"]):
            code = main([a.run] + g + cmd)
            if code:
                print(f"report: `{' '.join(cmd)}` failed ({code}); the page is made without those pictures")
    env = dict(os.environ, AR_PROJECT=ROOT)
    tool = os.path.join(ROOT, "..", "autoresearch", "tools", "run_report.py")
    extra = ["--extra-gops", "frozen DualBEV image heads (71 GFLOPs per frame as stated in training/feature_cache.py, unit not verified: if they are MACs the figure doubles; the BEVFusion heads of a dual chain are not counted)=71"]
    return subprocess.call([sys.executable, tool, a.run, "--note", REPORT_NOTE] + extra + (["--vs"] + r.vs if r.vs else [])
                           + (["--val-table", r.val_table] if r.val_table else []) + (["--train-table", r.train_table] if r.train_table else []), env=env)


def cmd_generalisation(a, rest) -> int:
    """the same evaluation on the validation and on the TRAIN split (n frames each), then the report page with the train-vs-val table.
    n = the n of the stored plain val evaluation (its card), else the largest n in by_n/, else 600. A stored val evaluation of that n is reused; a val
    evaluation is only run when the run has none (it would replace the plain files, so a run with a val evaluation of ANOTHER n is refused: pass --n of it,
    or evaluate a copy of the run folder). Val and train must come from the same checkpoint step and the same GT signature, else nothing is compared."""
    import re
    p = argparse.ArgumentParser(prog="generalisation", allow_abbrev=False)
    p.add_argument("--n", type=int, default=0)
    r = p.parse_args(rest)
    if a.control:
        print("generalisation: not with --control (the file names carry the control; compare the plain runs)")
        return 2
    ev = os.path.join(a.run, "eval")
    by_n = os.path.join(ev, "by_n")
    have = sorted(int(m.group(1)) for f in (os.listdir(by_n) if os.path.isdir(by_n) else []) for m in [re.fullmatch(r"table_n(\d+)\.txt", f)] if m)
    plain = (_read_json(os.path.join(ev, "card.json")) or {}).get("frames")
    n = r.n or plain or (have[-1] if have else 600)
    paths = {s: os.path.join(by_n, f"table{tag}_n{n}.txt") for s, tag in (("val", ""), ("train", "_train"))}
    if not os.path.exists(paths["val"]) and plain and plain != n:
        print(f"generalisation: this run's stored val evaluation has {plain} frames; evaluating val with n={n} would replace its plain files. Use --n {plain} or a copy of the run folder.")
        return 2
    for split in ("val", "train"):
        if os.path.exists(paths[split]):
            print(f"generalisation: {split} n={n} already evaluated, reused", flush=True)
            continue
        b = argparse.Namespace(**vars(a))
        b.split = split
        code = cmd_eval(b, ["--n", str(n)])
        if code:
            return code
    mv, mt = (_read_json(os.path.join(by_n, f"metrics{tag}_n{n}.json")) or {} for tag in ("", "_train"))
    step = lambda m: (m.get("metrics") or {}).get("info/step")  # noqa: E731
    if mv.get("gt_signature") != mt.get("gt_signature") or step(mv) != step(mt):
        print(f"generalisation: val and train do not match (GT {mv.get('gt_signature')} / {mt.get('gt_signature')}, step {step(mv)} / {step(mt)}): not compared; "
              "evaluate the one that is out of date again")
        return 2
    print(f"generalisation: val and train, {n} frames each, step {step(mv)}, GT {mv.get('gt_signature')} (the TEST split is not touched: it needs a logged row under a recorded decision)")
    return cmd_report(a, ["--no-pictures", "--val-table", paths["val"], "--train-table", paths["train"]])


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "compare":
        return cmd_compare(argv[1:])
    if argv and argv[0] == "front":
        from evaluation.front_compare import main as front_main
        return front_main(argv[1:])
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    p.add_argument("run")
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--split", default="val", choices=["val", "train"])
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--curves", action="store_true", help="also PR / ROC / calibration per layer and velocity regression (evaluation/bev_curves.py)")
    p.add_argument("--fusion", action="store_true", help="also boxes scored on the fusion area (detf: care boxes, detfp: + partial) and the coverage split of the complete GT set (cov), D37 / D38 (evaluation/bev_fusion_area.py)")
    p.add_argument("--velocity", action="store_true", help="also the velocity field per cell set (static, per class, moving) and at the GT box centres (field / object head / zero), evaluation/bev_velocity.py")
    p.add_argument("--control", default="", choices=["", "shuffle_radar", "zero_radar", "shuffle_cam", "zero_cam"])
    p.add_argument("--out", default=None)
    p.add_argument("--state", default="carry", choices=["carry", "reset", "warmup"])
    a, command = p.parse_known_args(argv)                      # the global options anywhere; the rest is the command
    a.command = command
    a.run = a.run.rstrip("/")
    if not a.command:
        p.error("a command is required (qual, quant, list, eval, compare, selftest-adapter)")
    if a.command[0] == "eval":
        return cmd_eval(a, a.command[1:])
    if a.command[0] == "time":
        return cmd_time(a, a.command[1:])
    if a.command[0] == "memory":
        return cmd_memory(a, a.command[1:])
    if a.command[0] == "report":
        return cmd_report(a, a.command[1:])
    if a.command[0] == "generalisation":
        return cmd_generalisation(a, a.command[1:])
    if a.command[0] == "modules":                              # CPU unless --device is given: the probe must not disturb a training on the GPU
        from evaluation import bev_modules
        return bev_modules.main([a.run, "--device", a.device if "--device" in argv else "cpu", "--split", a.split] + (["--out", a.out] if a.out else []) + a.command[1:])
    if a.command[0] == "timing":
        from evaluation import bev_timing
        return bev_timing.main([a.run, "--device", a.device] + (["--out", a.out] if a.out else []) + a.command[1:])
    from evalsuite.cli import main as suite_main
    out = a.out or os.path.join(a.run, "eval", "suite")
    pre = ["--adapter", "evaluation.bev_adapter:build", "--arg", f"run={a.run}", "--arg", f"device={a.device}", "--arg", f"workers={a.workers}",
           "--arg", f"split={a.split}", "--arg", f"control={a.control}", "--out", out, "--state", a.state]
    if a.ckpt_dir:
        pre += ["--arg", f"ckpt_dir={a.ckpt_dir}"]
    return suite_main(pre + a.command)


if __name__ == "__main__":
    raise SystemExit(main())
