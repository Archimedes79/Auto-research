"""Is this config startable? Static checks (no GPU, no data read, no training step), the smoke config, and the start lines.

    python training/startable.py <config.json> [--require-clean] [--expect-gt SIGNATURE]
    python training/startable.py <config.json> --write-smoke DIR     also write DIR/smoke_<name>.json (20 steps, evaluation and checkpoint
                                                                     every 10, one checkpoint, dt != 0 terms from step 1: the loss
                                                                     warm-up of the config is replaced); the end-to-end smoke is run.py
    python training/startable.py                                     self-test

The checks repeat what `train.build` / `check_resume` refuse and what a start needs: the file parses (an unknown key is an error),
`ModelConfig` / `LossConfig` / `model.check_config` accept it, the head layout flags of model and loss agree, the camera kind and its feature
caches fit (`model.CAMERAS`), `loss.fusion_area_only` is not combined with `model.refine` (the refiner builds its own box mask), every named path
exists (data root, caches, all pre-trained weights, `init_from`, `model.student_init` with a checkpoint in it), an existing run folder is not finished and its stored config equals this one
(`train.check_resume`), the configs that stop at the first step or after the model is built are refused (`motion_block` needs `n_cam` >= 2, `loss.dt_weights`
needs a weight near |dt| 0 and 0.5, `train_only` / `freeze` must name parameters: `freeze.plan` on a model built on the meta device, no weights),
`radar_aux` without `delta_slots` and `loss.w_obj_mov` without `model.object_moving` are noted as doing nothing, the disk has room, and the GT: the GtGenerator that would be imported (a pinned copy `<out_root>/<name>/GtGenerator`, else the live
`GtGenerator/`) must have the signature files of the version `init_from`'s run was trained on (`gt.json` + `GtGenerator/versions.csv`), or of
`--expect-gt`; the live GT signature is otherwise hashed from the source at start, so a changed GtGenerator silently means another GT. The uncommitted
files of `training/` are listed (the launcher's snapshot is a copy of the working tree, so the arms of a matched control can differ in code);
`--require-clean` makes them a problem. Findings are problems (exit 1) or notes (exit 0). Run from `bev_fusion/` in the WSL with the gcar
environment.
"""
import os as _os
import sys as _sys
_sys.path.insert(1, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "net"))  # net/ (a flat run snapshot holds netpath.py next to this file)
import netpath  # noqa: E402,F401
import argparse
import numbers
import ast
import csv
import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                   # bev_fusion/
sys.path.insert(0, HERE)

import freeze as freeze_mod                                     # noqa: E402
import losses                                                   # noqa: E402
import model as M                                               # noqa: E402
import train as T                                               # noqa: E402
from image_cache import KEY, build_settings, key_built, key_missing_scenes, older_built, older_key, older_missing_scenes, older_root      # noqa: E402
from student_encoder import newest_checkpoint                   # noqa: E402

SMOKE = dict(steps=20, warmup=5, eval_every=10, save_every=10, keep=1, log_every=5, keep_steps=[])      # keep_steps: the engine refuses a step above `steps` or off the save_every grid (t26 lists 6000 .. 36000)
SMOKE_LOSS = dict(warmup_hold=0.0, warmup_ramp=0.01)           # the dt != 0 terms and the aux heads run inside 20 steps
CKPT_GB = 0.46                                                 # one complete checkpoint (state + ema + Adam moments), measured 445.7 MB
CKPT_GB_STUDENT = 0.54                                         # with a trainable camera student (t26 smoke: 541,673,853 bytes)


def _abs(p, root):
    p = os.path.expanduser(p)
    return p if os.path.isabs(p) else os.path.join(root, p)


def _git(args, cwd, text=True):
    """stdout of a git call, None when git is missing or the call fails (not a repository, unknown ref)"""
    try:
        r = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=text, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def dirty(root):
    """uncommitted files of training/ (names relative to it) and of net/ (names "net/..."), renames with the new name, untracked folders listed file by file, or
    None when the state cannot be read (not a repository, no git)"""
    out = _git(["status", "--porcelain", "-z", "-uall", "--", "training", "net"], root)
    if out is None:
        return None
    names, parts = [], out.split("\0")
    i = 0
    while i < len(parts):
        e = parts[i]
        if len(e) > 3:
            path = e[3:]
            names.append(path.split("training/", 1)[-1] if "training/" in path else path[path.find("net/"):])
            if e[0] in "RC":                                     # -z: the old name of a rename / copy follows as its own entry
                i += 1
        i += 1
    return names


def gt_identity(folder, base_signature, repo):
    """(True / False / None, message): are the signature files of the GtGenerator in `folder` the ones of the released version whose generator
    signature is `base_signature` (GtGenerator/versions.csv names its git tag)? None = cannot be told here"""
    sig = str(base_signature).rsplit("_", 1)[-1]
    try:
        rows = list(csv.DictReader(open(os.path.join(repo, "GtGenerator", "versions.csv"), newline="", encoding="utf-8")))
    except OSError:
        return None, "GtGenerator/versions.csv is not readable: GT identity not checked"
    row = next((r for r in rows if r["signature"] == sig), None)
    if row is None:
        return None, f"GT signature {sig} is not a released GtGenerator version: GT identity not checked"
    tag = row["tag"]
    prefix = next((p for p in ("GtGenerator/", "autogt/", "bev_fusion/autogt/") if _git(["cat-file", "-e", f"{tag}:{p}nuscenes_gt.py"], repo) is not None), None)   # the folder had the name autogt/ up to v12 / l5
    if prefix is None:
        return None, f"tag {tag} not found: GT identity not checked"
    src = _git(["show", f"{tag}:{prefix}nuscenes_gt.py"], repo)
    m = re.search(r"SIGNATURE_FILES\s*=\s*(\([^)]*\))", src or "")
    if not m:
        return None, f"SIGNATURE_FILES not found in {tag}: GT identity not checked"
    norm = lambda b: b.replace(b"\r\n", b"\n")                  # noqa: E731 - a checkout with CRLF is the same source
    bad = []
    for f in ast.literal_eval(m.group(1)):
        want = _git(["show", f"{tag}:{prefix}{f}"], repo, text=False)
        try:
            have = open(os.path.join(folder, f), "rb").read()
        except OSError:
            have = None
        if want is None or have is None or norm(want) != norm(have):
            bad.append(f)
    where = os.path.relpath(folder, repo).replace(os.sep, "/")
    if bad:
        return False, (f"the GtGenerator that would be imported ({where}) differs from {row['version']} ({sig}) in {', '.join(bad)}: the GT would "
                       f"be another one (a cache rebuild, other numbers); pin {row['version']} as a copy at <out_root>/<name>/GtGenerator "
                       f"(`git archive {tag} {prefix.rstrip('/')} | tar -x -C <out_root>/<name>"
                       f"{' --strip-components=%d' % (prefix.count('/') - 1) if prefix.count('/') > 1 else ''}"
                       f"{' && mv <out_root>/<name>/autogt <out_root>/<name>/GtGenerator' if prefix.rstrip('/').endswith('autogt') else ''}`)")
    return True, f"GT: {where} has the signature files of {row['version']} ({sig})"


def freeze_findings(mc, train_only, freeze):
    """(problems, notes) of the freeze plan: `freeze.plan` on the model built on the meta device (names and requires_grad only, no weights, no memory);
    a prefix that matches no parameter, or a plan that freezes everything, would stop the run after the model is built"""
    if not (train_only or freeze):
        return [], []
    try:
        with torch.device("meta"):
            model = M.BEVFusionModel(dataclasses.replace(mc, pretrained="", trunk_pretrained="", lane_pretrained=""))
    except Exception as e:                                      # noqa: BLE001 - a model that cannot be built on meta: say so, do not refuse
        return [], [f"train_only / freeze not checked against the model ({type(e).__name__}: {e})"]
    try:
        plan = freeze_mod.plan(model, train_only, freeze)
    except (ValueError, TypeError) as e:
        return [f"train_only / freeze: {e}"], []
    return ([] if any(plan.values()) else ["train_only / freeze froze every parameter: nothing is left to train"]), []


def clip_split_findings(mc, split):
    """problems of `grad_clip_split` (a prefix that names no trainable parameter, two prefixes that overlap, a bare string) on the model built on the meta device: train.py
    would raise them only after the data are built"""
    if not split:
        return []
    try:
        with torch.device("meta"):
            model = M.BEVFusionModel(dataclasses.replace(mc, pretrained="", trunk_pretrained="", lane_pretrained=""))
    except Exception:                                           # noqa: BLE001 - freeze_findings already says a model cannot be built on meta
        return []
    try:
        T.clip_groups(model, split)
    except ValueError as e:
        return [str(e)]
    return []


def older_cache_findings(lane_images, cfg):
    """problems of the older-frame image cache (student_frames 'all'): <lane_images>/sweeps must hold a FINISHED build of this window (n_cam, cam_stride) that covers every scene of the config -
    a half-built cache fails at its first missing frame, perhaps only in the first evaluation, hours into the run"""
    scenes, built, key = set(cfg.train_scenes) | set(cfg.val_scenes), older_built(lane_images), older_key(cfg.n_cam, cfg.cam_stride)
    if not os.path.isdir(older_root(lane_images)):
        return [f"model.student_frames 'all' reads the older frames' images from {cfg.lane_images}/sweeps: it does not exist (python training/build_image_cache.py build-cache-older <config>)"]
    if not scenes:
        return ["model.student_frames 'all': the config names no scenes (a split), so the older-frame cache cannot be checked against them: use a split"]
    if key not in built:
        return [f"model.student_frames 'all': {cfg.lane_images}/sweeps has no finished build for {key} (BUILT.json lists {sorted(built) or 'nothing'}; python training/build_image_cache.py build-cache-older <config>)"]
    if built[key].get("settings") != build_settings():
        return [f"model.student_frames 'all': the build of {key} in {cfg.lane_images}/sweeps was made with {built[key].get('settings', 'no recorded settings (a marker of an older builder)')}, the builder now makes "
                f"{build_settings()}: build it again (python training/build_image_cache.py build-cache-older <config> in a new folder)"]
    missing = older_missing_scenes(lane_images, cfg.n_cam, cfg.cam_stride, scenes)
    return [f"model.student_frames 'all': the finished build of {key} in {cfg.lane_images}/sweeps lacks {len(missing)} of the config's {len(scenes)} scenes, e.g. {missing[:3]} "
            "(python training/build_image_cache.py build-cache-older <config> with this config)"] if missing else []


def key_cache_findings(lane_images, cfg):
    """problems of the key-frame image cache of a student_frames 'all' config: <lane_images>/BUILT.json must hold a FINISHED key-frame build that covers every scene of the config (the older-frame
    cache has its own marker). Other configs with lane_images are not asked: their caches were built before the marker existed"""
    scenes, built = set(cfg.train_scenes) | set(cfg.val_scenes), key_built(lane_images)
    if not scenes:
        return []                                                       # older_cache_findings says so
    if KEY not in built:
        return [f"model.student_frames 'all': {cfg.lane_images} has no finished key-frame build (BUILT.json lists {sorted(built) or 'nothing'}; python training/build_image_cache.py build-cache <config>: "
                "resumable, it writes the marker when every sample is through)"]
    if built[KEY].get("settings") != build_settings():
        return [f"model.student_frames 'all': the key frames in {cfg.lane_images} were made with {built[KEY].get('settings', 'no recorded settings (a marker of an older builder)')}, the builder now makes "
                f"{build_settings()}: build them again (python training/build_image_cache.py build-cache <config> in a new folder)"]
    missing = key_missing_scenes(lane_images, scenes)
    return [f"model.student_frames 'all': the finished key-frame build in {cfg.lane_images} lacks {len(missing)} of the config's {len(scenes)} scenes, e.g. {missing[:3]} "
            "(python training/build_image_cache.py build-cache <config> with this config)"] if missing else []


def read_resolved(path):
    """T.read_config of a config file; a "split" is replaced by its scene lists first (splits.resolve: what the Training class does before the trainer reads a config - the trainer itself refuses an
    unresolved one), so the check also runs on t20_base.json, t26_one.json and every other config that names a split"""
    import splits
    import tempfile
    raw = json.load(open(path))
    if "split" not in raw:
        return T.read_config(path)
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, os.path.basename(path))
        with open(out, "w") as fh:
            json.dump(splits.resolve(raw), fh)
        return T.read_config(out)


def keep_steps_problem(cfg):
    """the rule of trainkit.loop.fit for `keep_steps` (it refuses a bad list at its start): integers (no bool), each a multiple of save_every and not above steps"""
    bad = [k for k in (cfg.keep_steps or ()) if not (isinstance(k, numbers.Integral) and not isinstance(k, bool) and 0 < k <= cfg.steps and cfg.save_every and k % cfg.save_every == 0)]
    return [f"keep_steps {bad}: each must be an integer multiple of save_every ({cfg.save_every}) and not above steps ({cfg.steps})"] if bad else []


def checkpoint_gb(cfg):
    """the disk the checkpoints of a run take at their peak: the newest `keep`, every keep_steps one, and the one being written, each CKPT_GB (a trainable student's weights and moments make it CKPT_GB_STUDENT)"""
    one = CKPT_GB_STUDENT if cfg.model.get("camera_student") and cfg.model.get("student_trainable") else CKPT_GB
    return one, one * (cfg.keep + len(set(cfg.keep_steps or ())) + 1)


def check(path, root=ROOT, require_clean=False, expect_gt=""):
    """(problems, notes) of the config file `path`; nothing is read from the data, no model is built, nothing is trained"""
    problems, notes = [], []
    try:
        cfg = read_resolved(path)
    except Exception as e:                                      # noqa: BLE001 - the message is the finding
        return [f"config does not parse: {e}"], notes
    repo = os.path.dirname(root)
    mc = lc = None
    try:
        lc = losses.LossConfig(**cfg.loss)
        layout = {"centrelines": lc.centrelines, "dense_objects": lc.dense_objects, "radar_stationary": lc.radar_stationary,
                  "centreline_1m": lc.centreline_1m}
        mc = M.ModelConfig(**{**layout, **cfg.model})
        M.check_config(mc)
        for k, v in layout.items():
            if getattr(mc, k) != v:
                problems.append(f"model.{k} = {getattr(mc, k)} but loss.{k} = {v}: the head layout must match the loss")
        if cfg.balancing not in ("kendall", "fixed", "uwso"):
            problems.append(f"balancing {cfg.balancing!r}")
    except Exception as e:                                      # noqa: BLE001
        problems.append(f"model / loss fields: {e}")
        mc = None
    if mc is not None:
        cached = bool(cfg.feature_cache)
        if mc.camera_features != "dualbev" and not cached:
            problems.append(f"camera_features {mc.camera_features!r} needs a feature_cache")
        if ("feature_cache2" in {s.cache for s in M.CAMERAS[mc.camera_features]}) != bool(cfg.feature_cache2):
            problems.append(f"camera_features {mc.camera_features!r} and feature_cache2 do not match (the kinds with a second feature source need it, the others must not set it)")
        if mc.lane_branch and cached and not cfg.lane_images:
            problems.append("model.lane_branch with a feature_cache needs lane_images")
        problems += T.student_findings(cfg, mc)                 # camera_student: a feature cache, lane_images, the anchor's schedule and student
        if mc.camera_student and mc.student_frames == "all" and cfg.lane_images and os.path.isdir(_abs(cfg.lane_images, root)):
            problems += older_cache_findings(_abs(cfg.lane_images, root), cfg) + key_cache_findings(_abs(cfg.lane_images, root), cfg)
        si = _abs(mc.student_init, root) if mc.student_init else ""
        if si and os.path.exists(si) and newest_checkpoint(si) is None:
            problems.append(f"model.student_init: {mc.student_init} holds no checkpoint (ckpt_*.pt)")
        if mc.refine and lc.fusion_area_only:
            problems.append("loss.fusion_area_only with model.refine: the refiner builds its own box mask (losses.refine_gt_boxes) and ignores the option")
        if not cached and cfg.device == "cuda":
            notes.append("no feature_cache: the camera heads run live (slow, much more memory)")
        if mc.motion_block and cfg.n_cam < 2:
            problems.append(f"model.motion_block needs an older camera frame (motion_block.block_inputs): n_cam >= 2, is {cfg.n_cam}")
        for dt in (0.0, 0.5):                                   # the target times of the GT (train.py: 0 and +-0.5 s); the loss refuses a time without a weight within 0.1 s
            try:
                losses.dt_weight(dt, 1.0, lc)
            except (ValueError, TypeError) as e:
                problems.append(f"loss.dt_weights: {e}")
                break
        if cfg.radar_aux > 0 and mc.radar_kind != "delta_slots":
            notes.append(f"radar_aux {cfg.radar_aux} does nothing: only radar_kind 'delta_slots' returns the radar side loss's inputs (is {mc.radar_kind!r})")
        if "w_obj_mov" in cfg.loss and not mc.object_moving:
            notes.append("loss.w_obj_mov does nothing without model.object_moving (the head has no moving / static logit)")
        p_, n_ = freeze_findings(mc, cfg.train_only, cfg.freeze)
        problems += p_
        notes += n_
        problems += clip_split_findings(mc, cfg.grad_clip_split)
    named = [("data_root", cfg.data_root), ("feature_cache", cfg.feature_cache), ("feature_cache2", cfg.feature_cache2),
             ("gt_cache", cfg.gt_cache), ("radar_cache", cfg.radar_cache), ("lane_images", cfg.lane_images), ("init_from", cfg.init_from)]
    if mc is not None:
        named += [("model.pretrained", mc.pretrained), ("model.trunk_pretrained", mc.trunk_pretrained),
                  ("model.lane_pretrained", mc.lane_pretrained if mc.lane_branch == "res4" else ""), ("model.student_init", mc.student_init)]
    for key, p in named:
        if p and not os.path.exists(_abs(p, root)):
            problems.append(f"{key}: {p} does not exist")
    run = os.path.join(root, cfg.out_root, cfg.name)
    if os.path.isdir(run):
        cks = sorted(f for f in os.listdir(run) if f.startswith("ckpt_") and f.endswith(".pt"))
        if os.path.exists(os.path.join(run, ".done")):
            problems.append(f"{cfg.out_root}/{cfg.name} is a finished run (.done): pick another name")
        elif cks:
            notes.append(f"{cfg.out_root}/{cfg.name} has {len(cks)} checkpoint(s), the last {cks[-1]}: starting continues it (a resume, with a new snapshot)")
            try:
                T.check_resume(json.load(open(os.path.join(run, "config.json"))), cfg, cfg.name)
            except SystemExit as e:
                problems.append(f"resume refused: {e}")
            except OSError:
                notes.append("no stored config.json in the run folder: the resume check could not be made")
    free = shutil.disk_usage(root).free / 1e9
    problems += keep_steps_problem(cfg)
    one, need = checkpoint_gb(cfg)
    if free < need + 5:
        problems.append(f"free disk {free:.1f} GB on the run drive, the checkpoints need about {need:.1f} GB ({cfg.keep} kept + {len(set(cfg.keep_steps or ()))} keep_steps + the one being written, "
                        f"{one} GB each) plus margin")
    # GT identity: the version the base was trained on (init_from's run, else --expect-gt) against the GtGenerator that would be imported
    base = expect_gt
    if not base and cfg.init_from:
        try:
            base = json.load(open(os.path.join(os.path.dirname(_abs(cfg.init_from, root)), "gt.json")))["gt_signature"]
        except (OSError, KeyError, ValueError):
            notes.append("init_from's run has no readable gt.json: GT identity not checked")
    if base:
        pinned = os.path.join(run, "GtGenerator")
        ok, msg = gt_identity(pinned if os.path.isdir(pinned) else os.path.join(repo, "GtGenerator"), base, repo)
        (notes if ok is not False else problems).append(msg)
    else:
        notes.append("GT identity not checked (no init_from and no --expect-gt)")
    d = dirty(root)
    if d is None:
        notes.append("git state of training/ unknown (not a repository or no git)")
    elif d:
        (problems if require_clean else notes).append(
            f"the snapshot carries {len(d)} uncommitted file(s) of training/: {', '.join(d[:8])}{' ...' if len(d) > 8 else ''}")
    if cfg.save_every and cfg.steps % cfg.save_every and cfg.steps > cfg.save_every:        # save_every 0 = no periodic checkpoint (train.py accepts it)
        notes.append(f"steps {cfg.steps} is not a multiple of save_every {cfg.save_every}: the last checkpoint is written at the end only")
    return problems, notes


def smoke_config(path, out_dir):
    """Write the smoke twin of a config: the same model and loss except the loss warm-up (SMOKE_LOSS), 20 steps, no keep_steps, the warm-up of an lr group cut to the run's; returns its path."""
    d = json.load(open(path))
    base = d["name"]
    d.update(SMOKE, name="smoke_" + base)
    if d.get("lr_groups"):                                           # a group's own warm-up (t26: 3000) is longer than the run
        d["lr_groups"] = {k: dict(v, warmup=min(v["warmup"], SMOKE["warmup"])) if "warmup" in v else v for k, v in d["lr_groups"].items()}
    d["_about"] = (f"end-to-end smoke of {base}: the same model, loss and data, {SMOKE['steps']} steps, evaluation and a checkpoint every 10; "
                   f"the loss warm-up is replaced (loss.warmup_hold / warmup_ramp) so that the dt != 0 terms run")
    d["loss"] = dict(d.get("loss", {}), **SMOKE_LOSS)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, d["name"] + ".json")
    with open(out, "w") as fh:
        json.dump(d, fh, indent=1)
        fh.write("\n")
    return out


def start_lines(path, name, smoke=False):
    """the run.py call; the card text is not made up here: a real run needs --hyp / --pred / --rule from its card (the launcher accepts any
    text, so the placeholders must be replaced, not pasted); a smoke run is a diagnostic one and starts with --no-card"""
    rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
    rid = ("smoke." + name[len("smoke_"):]) if smoke else name.replace("_", ".", 1)    # t10_dual_long_s0 -> t10.dual_long_s0
    card = "--no-card" if smoke else "--hyp <hypothesis of the card> --pred <prediction> --rule <decision rule>"
    return (f"AR_PROJECT=../bev_fusion nohup python3 ../autoresearch/tools/run.py --id {rid} --gpu --split train {card} "
            f"-- bash tools/run_ar.sh {rel} > runs/{rid}.launcher.log 2>&1 &")


def _selftest():
    import tempfile
    with tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "temp")) as tmp:
        for d in ("data", "cache", "cache2", os.path.join("runs", "done_run"), os.path.join("runs", "base")):
            os.makedirs(os.path.join(tmp, d))
        open(os.path.join(tmp, "runs", "done_run", ".done"), "w").close()

        def write(**kw):
            c = {"name": "x", "out_root": "runs", "data_root": os.path.join(tmp, "data"), "feature_cache": os.path.join(tmp, "cache"),
                 "gt_cache": "", "model": {"pretrained": ""}, "loss": {}, "device": "cpu"}
            c.update(kw)
            p = os.path.join(tmp, c["name"] + ".json")
            json.dump(c, open(p, "w"))
            return p

        def has(findings, text):
            return any(text in x for x in findings)
        pr, nt = check(write(), tmp)
        assert pr == [] and has(nt, "GT identity not checked"), (pr, nt)
        bad = [(dict(bogus_key=1), "does not parse"), (dict(model={"no_such_field": 1}), "model / loss fields"),
               (dict(model={"fusion": "no_such_kind", "pretrained": ""}), "no_such_kind"), (dict(loss={"no_such_field": 1}), "model / loss fields"),
               (dict(model={"pretrained": "", "centrelines": True}), "head layout"), (dict(balancing="nope"), "balancing"),
               (dict(feature_cache=os.path.join(tmp, "missing")), "feature_cache"), (dict(init_from="runs/none.pt"), "init_from"),
               (dict(model={"pretrained": "weights/none.pth"}), "model.pretrained"),
               (dict(model={"camera_features": "dual", "pretrained": ""}), "do not match"),
               (dict(model={"camera_features": "bevfusion", "pretrained": ""}, feature_cache=""), "needs a feature_cache"),
               (dict(feature_cache2=os.path.join(tmp, "cache2")), "do not match"),
               (dict(model={"lane_branch": "rgb", "pretrained": ""}), "needs lane_images"),
               (dict(model={"refine": True, "pretrained": ""}, loss={"fusion_area_only": True}), "refiner builds its own box mask"),
               (dict(name="done_run"), "finished run"),
               # configs that stop at the first step or after the model is built (motion_block.block_inputs, losses.dt_weight, freeze.plan)
               (dict(model={"motion_block": True, "pretrained": ""}, n_cam=1), "needs an older camera frame"),
               (dict(loss={"dt_weights": [[0.0, 1.0]]}), "loss.dt_weights"),
               (dict(train_only=["fusoin."]), "freeze prefix 'fusoin.'"), (dict(freeze=["fusion.", "nosuchmodule"]), "freeze prefix 'nosuchmodule'"),
               (dict(train_only=["fusion."], freeze=["fusion."]), "froze every parameter")]
        os.makedirs(os.path.join(tmp, "student_run"))
        stud = {"camera_features": "dual", "camera_student": "resnet18", "student_trainable": True, "pretrained": ""}
        dual = dict(feature_cache2=os.path.join(tmp, "cache2"), lane_images=os.path.join(tmp, "data"))
        bad += [(dict(model=stud, feature_cache2=dual["feature_cache2"]), "needs lane_images"),                                                         # t22: the camera student
                (dict(model=dict(stud, student_init="runs/none"), **dual), "model.student_init: runs/none does not exist"),
                (dict(model=dict(stud, student_init="student_run"), **dual), "holds no checkpoint"),
                (dict(model=dict(stud, student_trainable=False), **dual), "frozen student"),
                (dict(model=dict(stud, student_trainable=False, student_init="student_run"), student_anchor=1.0, **dual), "student_anchor > 0"),
                (dict(model=dict(stud, student_frames="all"), n_cam=1, **dual), "n_cam >= 2"),                                                          # t26 student_frames 'all': the older frames' images
                (dict(model=dict(stud, student_frames="all"), n_cam=2, **dual), "build-cache-older"),                                                   # (no <lane_images>/sweeps yet)
                (dict(model=stud, img_jitter=-0.1, **dual), "img_jitter"), (dict(model=stud, img_jitter_from=1.0, **dual), "img_jitter_from"),         # t26 image jitter
                (dict(model={"pretrained": ""}, img_jitter=0.5), "needs model.camera_student"),
                (dict(steps=100, save_every=10, keep_steps=[10, 200]), "keep_steps [200]"), (dict(steps=100, save_every=10, keep_steps=[10, 15]), "keep_steps [15]"),     # t26 keep_steps: the engine's rule
                (dict(steps=100, save_every=10, keep_steps=[True]), "keep_steps [True]"), (dict(steps=100, save_every=10, keep_steps=[10.0]), "keep_steps [10.0]"),
                (dict(split="small", train_scenes=["scene-a"]), "does not parse"), (dict(split="no_such_split"), "no split")]
        for kw, text in bad:
            pr, _ = check(write(**kw), tmp)
            assert has(pr, text), (kw, text, pr)
        open(os.path.join(tmp, "student_run", "ckpt_00000001.pt"), "w").close()
        os.makedirs(os.path.join(tmp, "data", "sweeps"))
        older = dict(model=dict(stud, student_frames="all"), n_cam=2, train_scenes=["scene-a"], val_scenes=["scene-b"], **dual)
        pr, _ = check(write(**older), tmp)
        assert has(pr, "no finished build"), pr                                    # the folder is there, no BUILT.json: a half-built cache
        json.dump({"built": {older_key(2, T.TrainConfig.cam_stride): {"scenes": ["scene-a"], "settings": build_settings()}}}, open(os.path.join(tmp, "data", "sweeps", "BUILT.json"), "w"))
        pr, _ = check(write(**dict(older, cam_stride=T.TrainConfig.cam_stride + 1)), tmp)
        assert has(pr, "no finished build"), pr                                    # a finished build for another stride does not count
        pr, _ = check(write(**older), tmp)
        assert has(pr, "lacks 1 of the config's 2 scenes") and has(pr, "scene-b"), pr    # a finished build that does not cover every scene of the config
        pr, _ = check(write(**dict(older, train_scenes=[], val_scenes=[])), tmp)
        assert has(pr, "names no scenes"), pr
        key = older_key(2, T.TrainConfig.cam_stride)
        sweeps_marker = lambda **kw: json.dump({"built": {key: dict({"scenes": ["scene-b", "scene-a", "scene-c"], "settings": build_settings()}, **kw)}}, open(os.path.join(tmp, "data", "sweeps", "BUILT.json"), "w"))  # noqa: E731
        key_marker = lambda **kw: json.dump({"built": {KEY: dict({"scenes": ["scene-b", "scene-a", "scene-c"], "settings": build_settings()}, **kw)}}, open(os.path.join(tmp, "data", "BUILT.json"), "w"))            # noqa: E731
        sweeps_marker()
        pr, _ = check(write(**older), tmp)
        assert has(pr, "no finished key-frame build") and not has(pr, "sweeps"), pr                # the older frames are complete, the key frames have no marker (t26: 10,441 of 29,839 were missing)
        key_marker(scenes=["scene-a"])
        pr, _ = check(write(**older), tmp)
        assert has(pr, "key-frame build in") and has(pr, "lacks 1 of the config's 2 scenes"), pr    # a key-frame marker that does not cover every scene
        key_marker(settings=dict(build_settings(), jpeg_quality=50))
        pr, _ = check(write(**older), tmp)
        assert has(pr, "key frames in") and has(pr, "were made with"), pr                            # a cache made with other builder constants
        sweeps_marker(settings=dict(build_settings(), hw=[128, 352]))
        pr, _ = check(write(**older), tmp)
        assert has(pr, "was made with") and has(pr, "sweeps"), pr
        def without_settings(folder):                                                                # a marker of an older builder: no recorded settings
            f = os.path.join(folder, "BUILT.json")
            m = json.load(open(f))
            for e in m["built"].values():
                e.pop("settings", None)
            json.dump(m, open(f, "w"))
        sweeps_marker()
        without_settings(os.path.join(tmp, "data", "sweeps"))
        pr, _ = check(write(**older), tmp)
        assert has(pr, "no recorded settings") and has(pr, "sweeps"), pr                              # refused, not passed (N1)
        sweeps_marker()
        key_marker()
        without_settings(os.path.join(tmp, "data"))
        pr, _ = check(write(**older), tmp)
        assert has(pr, "no recorded settings") and has(pr, "key frames in"), pr
        key_marker()                                                                                 # both complete, with the builder's own constants
        for kw in (dict(model=stud, **dual), dict(model=dict(stud, student_init="student_run"), student_anchor=1.0, **dual),                          # a student that trains, with and without a start
                   older, dict(model=stud, img_jitter=0.5, img_jitter_from=0.25, **dual)):                                                          # ... with the older frames' cache, with image jitter
            pr, _ = check(write(**kw), tmp)
            assert pr == [], (kw, pr)
        # the same switches done right are no problem; a flag that does nothing is a note, not a refusal
        for kw in (dict(model={"motion_block": True, "pretrained": ""}), dict(train_only=["fusion."]), dict(save_every=0)):    # save_every 0 once raised ZeroDivisionError
            pr, _ = check(write(**kw), tmp)
            assert pr == [], (kw, pr)
        for kw, text, quiet in ((dict(radar_aux=0.5), "radar_aux 0.5 does nothing", dict(radar_aux=0.5, model={"radar_kind": "delta_slots", "c_rad": 64, "pretrained": ""})),
                                (dict(loss={"w_obj_mov": 2.0}), "w_obj_mov does nothing", dict(loss={"w_obj_mov": 2.0}, model={"object_moving": True, "pretrained": ""}))):
            pr, nt = check(write(**kw), tmp)
            assert pr == [] and has(nt, text), (kw, pr, nt)
            pr, nt = check(write(**quiet), tmp)
            assert pr == [] and not has(nt, "does nothing"), (quiet, pr, nt)
        # an existing run with a checkpoint: a changed config is refused by check_resume, an unchanged one is a note
        cfg_ok = T.read_config(write(name="base"))
        import dataclasses
        json.dump(dataclasses.asdict(cfg_ok), open(os.path.join(tmp, "runs", "base", "config.json"), "w"), default=str)
        open(os.path.join(tmp, "runs", "base", "ckpt_00000010.pt"), "w").close()
        pr, nt = check(write(name="base"), tmp)
        assert pr == [] and has(nt, "starting continues it"), (pr, nt)
        pr, _ = check(write(name="base", lr=0.5), tmp)
        assert has(pr, "resume refused"), pr
        real_usage = shutil.disk_usage
        shutil.disk_usage = lambda p: shutil._ntuple_diskusage(10, 9, 1 * 10 ** 9)
        try:
            pr, _ = check(write(), tmp)
        finally:
            shutil.disk_usage = real_usage
        assert has(pr, "free disk"), pr
        shutil.disk_usage = lambda p: shutil._ntuple_diskusage(100, 91, 8.5 * 10 ** 9)                  # 8.5 GB free: the budget counts keep_steps and a trainable student's checkpoint (t26: 0.54 GB each)
        try:
            t26ish = dict(model=stud, steps=2000, save_every=500, keep=2, keep_steps=[500, 1000, 1500, 2000], **dual)
            pr, _ = check(write(**t26ish), tmp)
            assert has(pr, "about 3.8 GB (2 kept + 4 keep_steps + the one being written, 0.54 GB each)"), pr
            pr, _ = check(write(**dict(t26ish, keep_steps=[], model=dict(stud, student_trainable=False, student_init="student_run"), student_anchor=0.0)), tmp)
            assert not has(pr, "free disk"), pr                                                       # 3 x 0.46 GB = 1.4 GB: fits
        finally:
            shutil.disk_usage = real_usage
        pr, _ = check(write(steps=100, save_every=10, keep_steps=[10, 100]), tmp)
        assert pr == [], pr                                                                          # a valid keep_steps list
        pr, _ = check(write(split="small"), tmp)
        assert pr == [], pr                                                                          # a config with a split is read as Training reads it (its lists filled in), not "does not parse"
        # relative paths are relative to the root, "~" expands
        os.makedirs(os.path.join(tmp, "rel_cache"))
        pr, _ = check(write(feature_cache="rel_cache"), tmp)
        assert pr == [], pr
        # uncommitted files of training/ and net/: renames, spaces, untracked folders; no repository = unknown
        repo = os.path.join(tmp, "repo")
        os.makedirs(os.path.join(repo, "training"))
        git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
        for n in ("a.py", "old.py"):
            open(os.path.join(repo, "training", n), "w").write("x = 1\n")
        for cmd in (["init", "-q"], ["add", "."], ["commit", "-q", "-m", "x"]):
            subprocess.run(git + cmd, cwd=repo, check=True, capture_output=True)
        assert dirty(repo) == [], dirty(repo)
        open(os.path.join(repo, "training", "a.py"), "w").write("x = 2\n")
        os.makedirs(os.path.join(repo, "training", "sub dir"))
        open(os.path.join(repo, "training", "sub dir", "c d.py"), "w").write("y = 1\n")
        subprocess.run(git + ["mv", "training/old.py", "training/new.py"], cwd=repo, check=True, capture_output=True)
        os.makedirs(os.path.join(repo, "net", "model"))
        open(os.path.join(repo, "net", "model", "m.py"), "w").write("z = 1\n")
        assert sorted(dirty(repo)) == ["a.py", "net/model/m.py", "new.py", "sub dir/c d.py"], dirty(repo)
        assert dirty(os.path.join(tmp, "data")) is None or isinstance(dirty(os.path.join(tmp, "data")), list)    # inside the outer repo or not: no crash
        # GT identity against a tagged released version
        os.makedirs(os.path.join(repo, "GtGenerator"))
        open(os.path.join(repo, "GtGenerator", "nuscenes_gt.py"), "w").write('SIGNATURE_FILES = ("a.py", "b.py")\n')
        for n in ("a.py", "b.py"):
            open(os.path.join(repo, "GtGenerator", n), "w").write(f"# {n}\n")
        open(os.path.join(repo, "GtGenerator", "versions.csv"), "w", newline="").write(
            "version,signature,tag,date,generator,what_changed\nv1,abc123,autogt/v1,2026-11-01,x,y\n")
        for cmd in (["add", "GtGenerator"], ["commit", "-q", "-m", "GtGenerator"], ["tag", "autogt/v1"]):
            subprocess.run(git + cmd, cwd=repo, check=True, capture_output=True)
        ok, msg = gt_identity(os.path.join(repo, "GtGenerator"), "0.5_4_abc123", repo)
        assert ok is True, msg
        open(os.path.join(repo, "GtGenerator", "b.py"), "w").write("# b changed\n")
        ok, msg = gt_identity(os.path.join(repo, "GtGenerator"), "0.5_4_abc123", repo)
        assert ok is False and "b.py" in msg and "pin v1" in msg, msg
        open(os.path.join(repo, "GtGenerator", "b.py"), "w", newline="").write("# b.py\r\n")             # a CRLF checkout is the same source
        assert gt_identity(os.path.join(repo, "GtGenerator"), "0.5_4_abc123", repo)[0] is True
        assert gt_identity(os.path.join(repo, "GtGenerator"), "0.5_4_zzz", repo)[0] is None
        # the real table: its header has the columns this reads, and the live GtGenerator is not v5 (the tag resolves, so the answer is False)
        real_repo = os.path.dirname(ROOT)
        assert {"version", "signature", "tag"} <= set(csv.DictReader(open(os.path.join(real_repo, "GtGenerator", "versions.csv"), newline="",
                                                                          encoding="utf-8")).fieldnames)
        if _git(["rev-parse", "--verify", "-q", "autogt/v5"], real_repo):                  # not in an archive without the repository
            ok, msg = gt_identity(os.path.join(real_repo, "GtGenerator"), "0.5_4_15efbde2d1", real_repo)
            assert ok is False and "differs from v5" in msg, (ok, msg)
        # the smoke twin keeps model and loss except the warm-up, is a valid config, and the start lines carry no made-up card
        s = smoke_config(write(name="base2", steps=6000, loss={"w_obj_vel": 1.0, "fusion_area_only": True, "warmup_hold": 0.2}),
                         os.path.join(tmp, "smk"))
        d = json.load(open(s))
        assert d["name"] == "smoke_base2" and d["steps"] == 20 and d["loss"]["fusion_area_only"] and d["loss"]["warmup_hold"] == 0.0
        assert "warm-up is replaced" in d["_about"] and T.read_config(s).steps == 20
        s2 = smoke_config(write(name="long", steps=36000, save_every=500, keep_steps=[6000, 12000, 36000], lr_groups={"student": {"lr": 3e-5, "warmup": 3000}, "motion_block": {"lr": 1e-4}}),
                          os.path.join(tmp, "smk"))
        c2 = T.read_config(s2)                                                                       # the engine refuses a keep_steps entry above `steps` / off the save_every grid at its start
        assert c2.steps == 20 and not c2.keep_steps and all(0 < k <= c2.steps and k % c2.save_every == 0 for k in c2.keep_steps), c2.keep_steps
        assert c2.lr_groups == {"student": {"lr": 3e-5, "warmup": 5}, "motion_block": {"lr": 1e-4}}, c2.lr_groups
        assert "--no-card" in start_lines(s, d["name"], smoke=True) and "--id smoke.base2 " in start_lines(s, d["name"], smoke=True)
        assert "<hypothesis of the card>" in start_lines(s, "t14_dc_ctl_s0") and "--id t14.dc_ctl_s0 " in start_lines(s, "t14_dc_ctl_s0")
        pr, _ = check(write(), tmp, require_clean=True)                                              # tmp lies in the outer repo's ignored temp/
        assert isinstance(pr, list)
    print("startable selftest passed: 36 kinds of bad config (incl. keep_steps, a split with lists, a marker without settings, the checkpoint disk budget; and the t22 camera student, its older frames' cache missing / half built / for other scenes, the image jitter) and a changed resume config are found, flags that do nothing are noted, relative paths and the disk check work, "
          "uncommitted files (rename, space, untracked folder) are listed, the GT identity is told against a tagged version (CRLF-safe), "
          "the smoke twin is a valid config, the start lines make up no card")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", nargs="?")
    ap.add_argument("--write-smoke", metavar="DIR")
    ap.add_argument("--require-clean", action="store_true", help="uncommitted files of training/ or net/ are a problem")
    ap.add_argument("--expect-gt", default="", help="GT signature the base was trained on (when the config has no init_from)")
    a = ap.parse_args()
    if not a.config:
        _selftest()
        sys.exit(0)
    cfg_path = os.path.abspath(a.config)
    problems, notes = check(cfg_path, ROOT, a.require_clean, a.expect_gt)
    for n in notes:
        print("note   :", n)
    for p in problems:
        print("PROBLEM:", p)
    name = json.load(open(cfg_path))["name"]
    if a.write_smoke:
        smk = smoke_config(cfg_path, os.path.abspath(a.write_smoke))
        print("smoke config:", smk)
        print("smoke start :", start_lines(smk, "smoke_" + name, smoke=True))
    print("start       :", start_lines(cfg_path, name))
    print("startable   :", "NO" if problems else "yes (static checks; the smoke run is the end-to-end proof)")
    sys.exit(1 if problems else 0)
