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
    time                                        ms per frame of the chain (idle GPU only) -> eval/latency.json
    memory [--clips 600]                        does the carried memory help? warm vs cold on the same frames -> eval/memory.json + txt
    compare RUN_A RUN_B [--tag _warm] [--n N] [--force]   paired scene bootstrap between two evaluated runs (b - a; runs evaluated
                                                before the suite: their scene_records pickles through evaluation/bev_legacy.py, only on
                                                the same frames); refused when the recorded GT signatures (evaluation: metrics / manifest,
                                                training: RUN/gt.json) differ, a warning when one is not recorded; --force overrides
Global options: --device cuda|cpu  --workers N (read-ahead processes, default 2)  --split val|train  --ckpt-dir DIR (moved checkpoints)
                --control shuffle_radar|zero_radar|shuffle_cam|zero_cam  --out DIR (default RUN/eval/suite)  --state carry|reset
Outputs: RUN/eval/suite/{qualitative, quant}; see autoresearch/evalsuite/EXTENSIONS.md chapter 5 for what each command writes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, ".."))                       # so that `autoresearch` can be located from any cwd
sys.path.insert(0, os.path.join(ROOT, "..", "autoresearch"))
sys.path.insert(0, ROOT)

HEADLINE_LAYERS = (("drivable_area", "iou@0"), ("drivable_area", "iou@0.5"), ("ped_crossing", "iou@0"), ("walkway", "iou@0"),
                   ("carpark_area", "iou@0"), ("lane_divider", "iou@0.5"), ("lane_divider", "iou@1"), ("road_divider", "iou@1"),
                   ("lane_divider:zc", "f1@0.5"), ("road_divider:zc", "f1@0.5"), ("lane_divider:zc", "field_mae"))


def tag_of(warm: bool, on_train: bool, control: str) -> str:
    return ("_warm" if warm else "") + ("_train" if on_train else "") + (f"_ctl_{control}" if control else "")


def to_legacy(table: dict, prefix_seg: str = "seg") -> dict:
    """suite table ('seg.fov/dt+0.00/drivable_area/all/all/iou@0', 'det.dt+0.00/vehicle/all/ap@2', ...) -> the flat names of the former
    evaluate_run metrics.json ('seg_fov/seg/dt+0.00/...', 'det/dt+0.00/...', 'detr/...', 'val_loss/total')"""
    flat = {}
    for k, v in table.items():
        m, rest = k.split(".", 1)
        if m == "seg":
            mask, r = rest.split("/", 1)
            flat[f"{prefix_seg}_{mask}/seg/{r}"] = v
        elif m in ("det", "detr"):
            flat[f"{m}/{rest}"] = v
        elif m == "loss":
            flat["val_loss/" + rest.split("/", 1)[1]] = v
    return flat


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


def train_gt(run: str):
    """the GT signature a run trained on (runs/<name>/gt.json, written by training/train.py since 2026-10-31); None = not recorded"""
    return (_read_json(os.path.join(run, "gt.json")) or {}).get("gt_signature")


def make_suite(a, prior: int = 0, state: str = None):
    from evalsuite import EvalSuite
    from evaluation.bev_adapter import build
    source, subject, task = build(a.run, device=a.device, split=a.split, workers=a.workers, ckpt_dir=a.ckpt_dir, control=a.control, prior=prior)
    out = a.out or os.path.join(a.run, "eval", "suite")
    return EvalSuite(source, subject, task, out=out, state=state or a.state)


def cmd_eval(a, rest) -> int:
    p = argparse.ArgumentParser(prog="eval")
    p.add_argument("--n", type=int, default=0, help="val frames spread over all scenes (0 = all)")
    p.add_argument("--warm", action="store_true", help="the memory carried along whole scenes (needs --n 0 or --scenes)")
    p.add_argument("--prior", type=int, default=0, help="train frames of the prior-map baseline (0 = off)")
    p.add_argument("--images", type=int, default=0)
    p.add_argument("--time", action="store_true")
    p.add_argument("--scenes", nargs="*", default=None)
    e = p.parse_args(rest)
    tag = tag_of(e.warm, a.split == "train", a.control)
    state = "carry" if e.warm else "reset"
    suite = make_suite(a, state=state)
    from evalsuite.sampling import Sampler
    frames = Sampler(suite.source, 0).spread(e.n, e.scenes) if e.n else None
    name = f"eval{tag}_n{e.n or 'all'}"
    t0 = time.time()
    from evaluation.bev_views import gt_boxes
    groups = list(suite.task.groups)
    det_rec, detr_rec = {}, {}

    def keep_boxes(sample, pred):                          # the boxes of every frame, for the legacy scene_records.pkl
        sc = sample.ref.scene
        for dt, o in pred.outputs["dt"].items():
            gb = gt_boxes(sample.reference[dt], groups)
            det_rec.setdefault(sc, {}).setdefault(dt, []).append((list(o["dets"]), gb))
            if o.get("detr") is not None:
                detr_rec.setdefault(sc, {}).setdefault(dt, []).append((list(o["detr"]), gb))
    store = suite.quant.run(scenes=e.scenes if not e.n else None, frames=frames, state=state, name=name, progress=50, on_frame=keep_boxes)
    flat = to_legacy(suite.quant.table(store))
    subj = suite.subject.describe()
    flat["info/step"], flat["info/n_samples"], flat["info/n_scenes"] = subj.get("step"), len(store.frames), len(store.scene_names())
    lat = None
    if e.time:
        lat = _latency(suite, a)
        flat.update({f"ms/{k}": v for k, v in lat.items() if isinstance(v, (int, float)) and k != "n"})
    if e.prior:
        psuite = make_suite(a, prior=e.prior, state="reset")
        pstore = psuite.quant.run(frames=frames or [f for s in suite.source.scenes() for f in suite.source.frames(s)], state="reset",
                                  name=f"prior{e.prior}_n{e.n or 'all'}", progress=50)
        flat.update(to_legacy(psuite.quant.table(pstore), prefix_seg="prior"))
        psuite.close()
    if e.images:
        for ref in Sampler(suite.source, 0).spread(e.images):
            suite.qual.frame(ref, ("overview", "outputs"), out_dir=os.path.join(a.run, "eval", "images"),
                             history="full" if e.warm else "none", state=state)
    out_dir = os.path.join(a.run, "eval")
    os.makedirs(out_dir, exist_ok=True)
    gt_sig = suite.source.describe().get("gt_signature")                 # the GT scored against (autogt is read live, not pinned)
    with open(os.path.join(out_dir, f"metrics{tag}.json"), "w") as f:
        json.dump({"run": a.run, "checkpoint": subj.get("checkpoint"), "split": a.split, "weights": "EMA",
                   "memory": "carried over the whole scene" if e.warm else "empty (cold)", "code": subj.get("code"),
                   "gt_signature": gt_sig, "train_gt_signature": train_gt(a.run),
                   "suite_run": os.path.relpath(store.path, a.run), "metrics": flat, "latency": lat}, f, indent=1, default=float)
    from evaluation import bev_legacy                                    # the n is in the name: an eval of another n never overwrites it
    rec_path = os.path.join(out_dir, f"scene_records{tag}_n{e.n or 'all'}.pkl")
    bev_legacy.write_scene_records(rec_path, suite.task.metrics()[0], store,
                                   det_rec, detr_rec, meta={"frames": [fr["key"] for fr in store.frames], "gt_signature": gt_sig})
    import shutil                                                        # the plain name too, always replaced: the temp readers (collect.py,
    shutil.copyfile(rec_path, os.path.join(out_dir, f"scene_records{tag}.pkl"))  # build_tables.py, ...) read it next to metrics<tag>.json
    table = headline_table(flat)
    with open(os.path.join(out_dir, f"table{tag}.txt"), "w") as f:
        f.write(table + "\n")
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
    cold = suite.quant.run(frames=frames, state="reset", name=f"memory_cold_n{m.clips}", progress=50)
    warm = suite.quant.run(frames=frames, state="warmup", name=f"memory_warm_n{m.clips}", progress=50)
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
        lines.append(f"  {k:62s} cold {v['a']:.4f}  warm {v['b']:.4f}  diff {v['diff']:+.4f}  95% [{v['lo']:+.4f}, {v['hi']:+.4f}]"
                     + ("  (excludes 0)" if v["lo"] > 0 or v["hi"] < 0 else ""))
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


def _task_only(run: str):
    """the Task of a run without loading its checkpoint (metrics definitions only)"""
    from evaluation import bev_adapter as B
    from evaluation.bev_task import BEVTask
    os.chdir(ROOT)
    B.setup_paths(run)
    ns = B._chain()
    with open(os.path.join(run, "config.json")) as f:
        cfg = ns.T.TrainConfig(**json.load(f))
    model, _, _ = ns.T.build(cfg)
    refine = bool(cfg.model.get("refine")) and hasattr(model, "boxes")
    return BEVTask(model.grid, model.spec, [g for g, _ in ns.NUSC_GROUPS], refine=refine)


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
    c = p.parse_args(argv)
    from evalsuite.quantitative import Quantitative
    name = f"eval{c.tag}_n{c.n}"
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
    task = _task_only(c.run_a)
    q = Quantitative(SimpleNamespace(out=os.path.join(c.run_a, "eval", "suite"), task=task))
    res = q.compare(os.path.join(c.run_a, "eval", "suite", "quant", name), os.path.join(c.run_b, "eval", "suite", "quant", name), c.boot, c.seed,
                    force=c.force, ignore={"source": ("feature_cache", "control")})   # the arms may differ in what the model is fed (t8: features)
    print(f"paired scene bootstrap ({c.boot} resamples), b - a; a = {c.run_a}, b = {c.run_b}; val; {res['n_common_frames']} common frames")
    for k, v in res["metrics"].items():
        print(f"{k:62s} a {v['a']:.4f}  b {v['b']:.4f}  b-a {v['diff']:+.4f}  95% [{v['lo']:+.4f}, {v['hi']:+.4f}]"
              + ("  (excludes 0)" if v["lo"] > 0 or v["hi"] < 0 else ""))
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "compare":
        return cmd_compare(argv[1:])
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    p.add_argument("run")
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--split", default="val", choices=["val", "train"])
    p.add_argument("--ckpt-dir", default=None)
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
    from evalsuite.cli import main as suite_main
    out = a.out or os.path.join(a.run, "eval", "suite")
    pre = ["--adapter", "evaluation.bev_adapter:build", "--arg", f"run={a.run}", "--arg", f"device={a.device}", "--arg", f"workers={a.workers}",
           "--arg", f"split={a.split}", "--arg", f"control={a.control}", "--out", out, "--state", a.state]
    if a.ckpt_dir:
        pre += ["--arg", f"ckpt_dir={a.ckpt_dir}"]
    return suite_main(pre + a.command)


if __name__ == "__main__":
    raise SystemExit(main())
