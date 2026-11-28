"""Trainer of the camera + radar BEV chain on nuScenes: the BEV side of the engine `autoresearch/trainkit` (EMA, schedule, optimiser groups and step, AMP,
checkpoints and resume, the stop protocol, the log files are the engine's).

    python -P -m bev_fusion.training.train <config.json>      fields of TrainConfig; unknown keys are an error. Works in the problem's data folder
                                                              (net/netpath.SITE): the config names its caches and weights relative to it

What is here: the config of a run (the network's fields of net/chain.py ChainConfig plus the trainer's), the data loaders, one training step = a clip of
consecutive keyframes through `chain.forward_loss` with the model state carried from keyframe to keyframe (`run_clip`), the validation under the EMA weights,
the freeze plan, the probes, the record of the ground truth a run used. The model, the loss, the train-only loss terms and the forward of a batch are net/
(chain.py); the evaluation imports them from there, never this file.
Sequence training (`seq_len` > 1): clips of consecutive keyframes of one scene (`batches.ClipDataset`), the world memory of the temporal fusion carried from
keyframe to keyframe with the ego step between them (`batches.clip_step`), detached at every keyframe or, with `seq_bptt`, kept in the graph through the clip;
the first keyframe of a clip starts cold, the later ones warm, so both cases are trained. With `seq_len` = 1 every keyframe is independent. Validation of clips
reports the last keyframe warm
and cold (`total_last_warm` / `total_last_cold`, the same frames) on a fixed subset of `eval_batches * batch_size` samples spread over all val scenes.
BatchNorm statistics come from the micro-batch (`batch_size`, not the effective batch `batch_size * accum`).
"""
import contextlib
import functools
import json
import os
import sys
from dataclasses import dataclass

import torch

from bev_fusion.net import netpath
from bev_fusion.net.chain import REMOVED, ChainConfig, build, external, forward_loss, read_raw
from bev_fusion.net.prep.augment import draw_jitter, flip_batch, jitter_seed
from bev_fusion.net.prep.batches import ClipDataset, clip_step, collate_clips, make_datasets, val_subset
from bev_fusion.net.prep.feature_cache import collate_cached
from bev_fusion.net.prep.nuscenes_dataset import collate
from bev_fusion.training import freeze as freeze_mod
from trainkit.config import Config
from trainkit.data import EpochLoader
from trainkit.loop import Hooks, fit
from trainkit.optim import clip_groups, lr_group_params
from trainkit.probe import ModuleProbe, StepProfiler, StepTimer, batch_tokens, dump_incident
from trainkit.util import detach_tree, prepare_run_dir, resolve_device, seed_all

# a resume may change these on top of the engine's (paths, workers, the cadence of logging and saving, the switches that only look); every other difference
# is refused. The top-level options that left the code are among them: a checkpoint written before names them, today's config cannot (a removed option of
# `model` / `loss` stays in its dict, chain.drop_removed, so the dicts of such a run still compare equal)
RESUME_FREE = Config.RESUME_FREE + ("data_root", "feature_cache", "gt_cache", "feature_cache2", "radar_cache", "lane_images", "workers", "eval_batches", "grad_diag_every",
                                    "probe", "pin_memory", "prefetch_factor", "profile_steps") + tuple(REMOVED[""])
# switches a checkpoint from before them lacks: turning one on in the middle of a run changes the numbers (a freeze plan, the speed flags, the split clip,
# the lr groups)
RESUME_SWITCHES = ("train_only", "freeze", "cudnn_benchmark", "tf32", "grad_clip_split", "lr_groups", "student_anchor", "img_jitter")


@dataclass
class TrainConfig(ChainConfig, Config):
    """the fields of the engine (trainkit.config.Config: name, out_root, steps, lr, warmup, accum, amp, ema, clipping, lr_groups, logging / saving cadence,
    seed, init_from, device), the network's (chain.ChainConfig: data, caches, model, loss, the train-only loss terms) and the trainer's own, below"""
    RESUME_FREE = RESUME_FREE                    # not fields: the rules of a resume (trainkit.checkpoint.check_resume); BF_RESUME_ANY=1 overrides the refusal
    RESUME_SWITCHES = RESUME_SWITCHES
    RESUME_ENV = "BF_RESUME_ANY"
    batch_size: int = 2
    workers: int = 2
    # strength of the photometric jitter of the camera student's input in training (needs model.camera_student): brightness, contrast, saturation +-0.4 x it,
    # gamma 1.25 ** (+-it), gaussian noise 0.02 x it (augment.jitter_images); one random draw per sample (a clip: one), the same for every frame of it, a
    # function of (seed, step, micro-batch) so a resume repeats it; the anchor keeps comparing with the teachers' outputs of the clean image. 0 = off, bit for
    # bit
    img_jitter: float = 0.0
    # the fraction of the steps (step / steps) from which the jitter is on: until the anchor has reached its floor the student's input is the clean image the
    # cached targets belong to
    img_jitter_from: float = 0.0
    grad_diag_every: int = 0                     # log the cosines of the task-group gradients at the BEV feature every N steps
    # backpropagate through the carried memory over the clip (graphs of all keyframes stay in memory); False = the state is detached at every keyframe
    seq_bptt: bool = False
    # probability that a sample (a whole clip) is mirrored left-right in training (net/prep/augment.py: BEV, radar, ego motion and every target); 0 = off
    flip_aug: float = 0.0
    eval_batches: int = 50
    train_only: tuple = ()                       # module prefixes that train, everything else is frozen (freeze.py; () = everything trains)
    # module prefixes that are frozen (freeze.py); with train_only it takes from it. Not in RESUME_FREE: a resume needs the same plan
    freeze: tuple = ()
    prefetch_factor: int = 0                     # batches each loader worker prepares ahead (0 = the torch default, 2); only with workers > 0
    # torch.backends.cudnn.benchmark: the fastest convolution algorithm per input shape (shapes are static here); the last bits may differ
    cudnn_benchmark: bool = False
    # torch.set_float32_matmul_precision("high"): TF32 for fp32 matrix products (torch already runs convolutions in TF32 by default); changes the numbers
    # slightly
    tf32: bool = False
    # > 0: torch.profiler over that many optimiser steps (at most 8: 30 steps held 22 GB of host RAM) after 10 warm-up steps; writes profile.txt into the run
    # folder (trainkit.probe.StepProfiler)
    profile_steps: int = 0
    # log grad.<module> / upd.<module> (gradient norm, update-to-weight ratio) and t_data / t_step in the rows of log_every, dump incident_<step>.json on a
    # skipped step (trainkit/probe.py; costs one copy of the trainable weights per logged step)
    probe: bool = False


def read_config(path):
    """TrainConfig of a config file (chain.read_raw: a "split" becomes its scene lists, an option that left the code is dropped at its off value and refused
    at any other); keys starting with "_" are comments for the reader (e.g. "_about"), every other unknown key is an error"""
    return TrainConfig(**{k: v for k, v in read_raw(path).items() if not k.startswith("_")})


def jitter_findings(cfg, mcfg):
    """what the image jitter needs of the config, as messages (train raises them)"""
    out = []
    if not 0 <= cfg.img_jitter <= 2 or not 0 <= cfg.img_jitter_from < 1:
        out.append(f"img_jitter {cfg.img_jitter} (0 = off .. 2) and img_jitter_from {cfg.img_jitter_from} (0 <= from < 1, a fraction of the steps)")
    if cfg.img_jitter > 0 and not mcfg.camera_student:
        out.append("img_jitter jitters the camera student's input: it needs model.camera_student")
    return out


def make_loader(ds, cfg, train, workers=None, generator=None):
    """workers: default cfg.workers. The validation loader of train() gets at most 2 workers that are not kept alive between evaluations,
    and its own generator: a worker that builds a missing GT loads the map files, and t7_rgb_s1 died of ENOMEM in validation worker 4 at
    its first evaluation (2026-10-31); with the generator an evaluation no longer draws from the training's random stream.
    The training loader (train, no generator given) is a trainkit EpochLoader: the order of a pass is a function of (cfg.seed, epoch), so a resumed run
    sees the pass it was in from the batch it stopped at (and the order of a new run is no longer drawn from the global random stream)."""
    fn = collate_cached if cfg.feature_cache else collate
    if cfg.seq_len > 1:
        fn = functools.partial(collate_clips, fn=fn)
    n = cfg.workers if workers is None else workers
    extra = {}
    if cfg.pin_memory and torch.cuda.is_available():
        extra["pin_memory"] = True
    if cfg.prefetch_factor > 0 and n > 0:
        extra["prefetch_factor"] = cfg.prefetch_factor
    common = dict(batch_size=cfg.batch_size, num_workers=n, collate_fn=fn, drop_last=train, persistent_workers=train and n > 0, **extra)
    if train and generator is None:
        return EpochLoader(ds, seed=cfg.seed, **common)
    return torch.utils.data.DataLoader(ds, shuffle=train, generator=generator, **common)


# ------------------------------------------------------------------------------------------------ loop
def _f(v):
    return float(v.detach()) if torch.is_tensor(v) else float(v)


def _pack(out, selected):
    """snapshot of the logged scalar terms of ONE keyframe: (order, packed). The tensors are stacked NOW into one device tensor (a term that is a view of a
    parameter, which a later keyframe changes in place, must be read as it is at this point; the stack is a copy), numbers are kept. Nothing is read back here:
    the old code called float() per
    term, and every call waited for the GPU, which was still busy with the backward of this keyframe, so the host could not queue the next keyframe meanwhile.
    The terms are widened to float64 (exact for float16 / bfloat16 / float32 / float64, as float(v) was)"""
    order, tens = [], []
    for k, v in out.items():
        if selected(k):
            if torch.is_tensor(v):
                order.append((k, True, len(tens)))
                tens.append(v.detach().double().reshape(()))
            else:
                order.append((k, False, float(v)))
    return order, (torch.stack(tens) if tens else None)


def _unpack(kf, L):
    """-> (acc, totals) of the packed keyframes with ONE device-to-host transfer. Bit-identical to the per-term floats: the same float32 values widened to
    Python floats, summed in the same order (keyframe by keyframe, term by term)"""
    packs = [p for _, p in kf if p is not None]
    flat = torch.cat(packs).cpu().tolist() if packs else []
    acc, totals, pos = {}, [], 0
    for order, p in kf:
        base = pos
        pos += 0 if p is None else p.numel()
        for k, is_t, x in order:
            val = flat[base + x] if is_t else x
            acc[k] = acc.get(k, 0.0) + val / L
            if k == "total":
                totals.append(val)
    return acc, totals


# single terms logged (train and val) next to the groups:
LOGGED_TERMS = ("objhead.vel@+0.0", "objhead.mov@+0.0", "motion@+0.0", "radar_aux", "radar_kept", "student_anchor")
                                                                             # the velocity terms (t11), hidden in group.objects / motion


def run_clip(model, loss, clip, step_frac, device, cfg, diag=False, train_terms=True, backward=False, augment=False, scaler=None, step=0, micro=0):
    """One clip = a list of batches (consecutive keyframes of the same scenes; a single batch = a clip of length 1). The
    model state (world memory) is carried from keyframe to keyframe with the ego step between them: detached at every
    keyframe, or with cfg.seq_bptt kept in the graph through the clip. backward: backpropagate the mean loss of the clip
    divided by cfg.accum, right after each keyframe (with seq_bptt the earlier graphs are retained for the state path of
    the later keyframes, so the gradient is that of the clip's mean loss). scaler: the GradScaler of amp "fp16": the loss is scaled before the backward (the
    gradients stay scaled until optimizer_step unscales them); the logged terms are the unscaled ones. augment: the mirror (cfg.flip_aug) and, from step_frac =
    cfg.img_jitter_from on, the photometric jitter of the student's image (cfg.img_jitter: batch["img_jitter"], one draw per clip of this micro-batch, drawn
    from (cfg.seed, step, micro) = the optimiser
    step and the micro-batch inside it).
    -> (dict of the scalar terms averaged over the keyframes, finite flag, diag of the last keyframe or None, list of the
    per-keyframe totals)."""
    L, state, prev, kf, dg = len(clip), None, None, [], None
    selected = lambda k: k == "total" or k == "consistency" or k.startswith(("group.", "s.")) or k in LOGGED_TERMS          # noqa: E731
    if augment and cfg.flip_aug > 0:                             # mirror a sample in ALL its keyframes (same memory frame)
        flip = torch.rand(len(clip[0]["tokens"])) < cfg.flip_aug
        for batch in clip:
            flip_batch(batch, flip)
    jitter = augment and cfg.img_jitter > 0 and step_frac >= cfg.img_jitter_from
    for t, batch in enumerate(clip):
        # the same draw for every key frame of a clip (one scene), the pixel noise differs per key frame
        if jitter:
            batch["img_jitter"] = draw_jitter(len(batch["tokens"]), cfg.img_jitter, jitter_seed(cfg.seed, step, micro), t)
        step_ego = clip_step(prev, device) if prev is not None else None
        out, res, _ = forward_loss(model, loss, batch, step_frac, device, cfg, diag=diag and t == L - 1, train_terms=train_terms, state=state, step=step_ego)
        if not torch.isfinite(out["total"]):
            acc, totals = _unpack(kf, L)
            return acc, False, dg, totals
        kf.append(_pack(out, selected))                           # before the backward, which changes the Kendall weights in place (clamp_)
        if backward:           # backward at once: the loss of a later keyframe changes the Kendall weights in place (clamp_)
            mean_loss = out["total"] / (L * cfg.accum)
            # bptt: the graph is still needed for the state path
            (mean_loss if scaler is None else scaler.scale(mean_loss)).backward(retain_graph=cfg.seq_bptt and t < L - 1)
        if "diag" in out:
            dg = out["diag"]
        state = res["state"] if cfg.seq_bptt else detach_tree(res["state"])
        prev = batch
    acc, totals = _unpack(kf, L)                                  # the only synchronisation of the clip besides the finite check of each keyframe
    return acc, True, dg, totals


def evaluate(model, loss, ema, loader, n_batches, device, cfg, eval_fn=None):
    """Validation loss (and groups) under the EMA weights, decoded like in training (cfg.gap_ego), without the train-only
    terms; eval_fn(model, loader) -> dict is an optional metrics hook. With clips (seq_len > 1) the state is carried through
    the clip, and the LAST keyframe is also run cold (no memory): total_last_warm / total_last_cold are the same frames with
    and without the carried memory (paired). A batch with a non-finite loss is left out of the mean and counted: the result carries `n` (batches in the mean)
    and `n_skipped` (logged as val.n, val.n_skipped); when none is finite the loss totals are NaN (never 0 or missing: a diverged model must not read as a good
    one)."""
    tot, n, skipped = {}, 0, 0
    with ema.applied(model):
        with torch.no_grad():
            for i, item in enumerate(loader):
                if i >= n_batches:
                    break
                clip = item if isinstance(item, list) else [item]
                out, ok, _, totals = run_clip(model, loss, clip, 1.0, device, cfg, train_terms=False)
                if not ok:
                    skipped += 1
                    continue
                for k, v in out.items():
                    if k == "total" or k.startswith("group.") or k in LOGGED_TERMS:
                        tot[k] = tot.get(k, 0.0) + v
                if len(clip) > 1:
                    cold, _, _ = forward_loss(model, loss, clip[-1], 1.0, device, cfg, train_terms=False)
                    tot["total_last_warm"] = tot.get("total_last_warm", 0.0) + totals[-1]
                    tot["total_last_cold"] = tot.get("total_last_cold", 0.0) + _f(cold["total"])
                n += 1
        res = {k: v / n for k, v in tot.items()} if n else {"total": float("nan")}
        res.update(n=n, n_skipped=skipped)
        if skipped:
            print(f"WARNING: validation left out {skipped} of {n + skipped} batches with a non-finite loss", flush=True)
        if eval_fn is not None:
            res.update(eval_fn(model, loader))
    return res


def record_gt(out_dir, datasets):
    """runs/<name>/gt.json = {"gt_signature": ...}: the identity of the GT the run trains and validates on (FrontRadarDataset.gt_signature:
    grid res, supersampling, GtGenerator source + flags). A pinned code snapshot does not pin the GT (GtGenerator is read live), so this file is
    the record of it (net/prep/splits.py manifest reads it). A resume on another GT keeps the earlier signatures and warns. -> the signature(s)"""
    sigs = sorted({d.gt_signature() for d in datasets if hasattr(d, "gt_signature")})
    if not sigs:
        return None
    sig = sigs[0] if len(sigs) == 1 else sigs
    path = os.path.join(out_dir, "gt.json")
    # D29: the human key of the GT version (v5, ...) next to the signature; none for an unreleased source
    try:
        from GtGenerator import version_of_key
        ver = [version_of_key(s) for s in sigs]
        ver = None if None in ver else (ver[0] if len(ver) == 1 else ver)
    except ImportError:
        ver = None
    rec = {"gt_signature": sig, **({"gt_version": ver} if ver else {})}
    if not os.path.exists(path) and any(f.startswith("ckpt_") and f.endswith(".pt") for f in os.listdir(out_dir)):
        rec["earlier"] = ["unknown: the run has checkpoints from before gt.json existed"]   # a resume of a run started before 2026-10-31
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        rec["earlier"] = list(old.get("earlier", []))
        if old.get("gt_signature") != sig:
            rec["earlier"].append(old.get("gt_signature"))
            print(f"WARNING: the GT changed since this run started ({old.get('gt_signature')} -> {sig}): the run mixes two GT versions",
                  flush=True)
        if not rec["earlier"]:
            del rec["earlier"]
    with open(path + ".tmp", "w") as f:
        json.dump(rec, f, indent=1)
    os.replace(path + ".tmp", path)
    return sig


def check_grids(model, *datasets):
    """The model and the GT of its datasets must share one cell grid (extents and resolution). `cfg.model["grid_kwargs"]` changes the
    model's grid only; the datasets are built with the default front grid, so another model grid would train on labels cut for
    another area, with the same shapes and no error."""
    g = model.grid
    for ds in datasets:
        d = getattr(ds, "grid", None) or getattr(getattr(ds, "base", None), "grid", None)
        if d is not None and (d.x_min, d.x_max, d.y_min, d.y_max, d.res) != (g.x_min, g.x_max, g.y_min, g.y_max, g.res):
            raise ValueError(f"the model grid x {g.x_min}..{g.x_max} y {g.y_min}..{g.y_max} res {g.res} is not the grid of the GT "
                             f"x {d.x_min}..{d.x_max} y {d.y_min}..{d.y_max} res {d.res} (make_datasets builds the default grid)")


def apply_speed_flags(cfg):
    """the opt-in speed switches of the config; all off by default and then nothing is touched. They are process-wide and set once at the start of train() (an
    eval_fn in the same process runs with them). -> {name: value} of what was set"""
    done = {}
    if cfg.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        done["cudnn.benchmark"] = True
    if cfg.tf32:
        torch.set_float32_matmul_precision("high")
        done["tf32"] = "high"
    return done


class ProbeHooks(Hooks):
    """trainkit/probe.py on the engine's hooks: the gradient norm and update ratio per module on the logged steps (ModuleProbe), the wait for the data and the
    length of a step (StepTimer),
    torch.profiler over a window of steps (StepProfiler) and incident dumps for a skipped step (dump_incident); each one only if the config asks for it
    (TrainConfig.probe, profile_steps). With both off every method does nothing, the loop's numbers are the same"""

    def __init__(self, model, out_dir, cfg, device):
        self.out_dir = out_dir
        self.probe, self.timer = (ModuleProbe(model, every=cfg.log_every), StepTimer()) if cfg.probe else (None, None)
        self.prof = StepProfiler(out_dir, cfg.profile_steps, device=device) if cfg.profile_steps > 0 else None
        self.step_ids = []

    def begin_step(self, n):
        self.step_ids = []
        return self.probe.begin(n) if self.probe is not None else None      # a StepProbe only on a step that is logged

    def data_wait(self):
        return self.timer.data() if self.timer is not None else contextlib.nullcontext()

    def batch(self, item):
        if self.probe is not None:
            self.step_ids.append(batch_tokens(item))                        # the samples of every micro-batch of the step, for an incident dump

    def skipped(self, step, what, gn, terms, step_frac):
        # what a replay needs: the step and the samples of the micro-batches so far (the last one is the one that failed)
        if self.probe is not None:
            dump_incident(self.out_dir, step, "nonfinite", batch_ids=[t for ids in self.step_ids for t in ids], losses=terms,
                          extra={"what": what, "micro": len(self.step_ids), "micro_batches": self.step_ids, "step_frac": step_frac,
                                 "grad_norm": None if gn is None else float(gn)})

    def end_step(self):
        row = self.timer.row() if self.timer is not None else {}
        if self.prof is not None:
            self.prof.tick()
        return row

    def close(self):
        if self.prof is not None:
            self.prof.close()


def train(cfg, eval_fn=None, datasets=None):
    """the BEV training: builds the network, the loss and the data, then hands the loop to the engine (trainkit.loop.fit) with run_clip as the step and evaluate
    as the validation.
    -> (model, loss, EMA, the last logged row)"""
    seed_all(cfg.seed)
    device = resolve_device(cfg.device)
    speed = apply_speed_flags(cfg)
    if speed:
        print("speed flags:", speed, flush=True)
    out_dir = prepare_run_dir(cfg)
    model, loss, _ = build(cfg)
    problems = jitter_findings(cfg, model.cfg)
    if problems:
        raise ValueError("; ".join(problems))
    model, loss = model.to(device), loss.to(device)
    if cfg.train_only or cfg.freeze:                            # one module trains, the rest is frozen (freeze.py): before the optimiser is built
        freeze_mod.apply(model, cfg.train_only, cfg.freeze)
        print("freeze:", freeze_mod.signature(cfg.train_only, cfg.freeze), flush=True)
        print(freeze_mod.report(model), flush=True)
    clip_groups(model, cfg.grad_clip_split)                     # a typo in a prefix fails here, in seconds, not after the data are built
    if cfg.lr_groups:
        lr_group_params(model, cfg.lr_groups)                   # the same for lr_groups (prefix, overlap, values)
    train_ds, val_ds = datasets or make_datasets(cfg)
    check_grids(model, train_ds, val_ds)
    record_gt(out_dir, (train_ds, val_ds))                      # the GT is not pinned by the code snapshot: record which one is used
    if cfg.seq_len > 1:                                         # clips of consecutive keyframes, the memory carried through
        train_ds, val_ds = ClipDataset(train_ds, cfg.seq_len), ClipDataset(val_ds, cfg.seq_len)
    train_loader = make_loader(train_ds, cfg, True)
    val_loader = make_loader(val_subset(val_ds, cfg.eval_batches * cfg.batch_size), cfg, False,      # fixed, spread over all scenes
                             workers=min(cfg.workers, 2), generator=torch.Generator().manual_seed(cfg.seed))
    if cfg.amp == "fp16" and cfg.grad_diag_every:
        print("WARNING: amp fp16: grad_diag_every is ignored (the group gradients would be taken unscaled and underflow)", flush=True)

    def step_fn(item, info):                                    # one micro-batch (a clip of keyframes): forward and backward
        # not under fp16: the group gradients would be taken unscaled and underflow
        diag = bool(cfg.grad_diag_every) and info.scaler is None and (info.step + 1) % cfg.grad_diag_every == 0
        out, ok, dg, _ = run_clip(model, loss, item if isinstance(item, list) else [item], info.frac, device, cfg,
                                  diag=diag and info.micro == 0, backward=True, augment=True, scaler=info.scaler, step=info.step, micro=info.micro)
        return out, ok, ({"diag": dg} if dg else None)

    return fit(cfg, model, loss, train_loader, step_fn, out_dir, device,
               evaluate=lambda ema: evaluate(model, loss, ema, val_loader, 10 ** 9, device, cfg, eval_fn),
               hooks=ProbeHooks(model, out_dir, cfg, device),
               frozen=freeze_mod.frozen_keys(model) if (cfg.train_only or cfg.freeze) else (), external=external(model))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    cfg_ = read_config(os.path.abspath(sys.argv[1]))
    os.chdir(netpath.SITE)                                      # the data folder: the config names its files relative to it
    train(cfg_)
