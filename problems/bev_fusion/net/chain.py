"""The network of a run, for the trainer and the evaluation alike (neither imports the other): its config, the model and the loss built from it, the
weights of a checkpoint, and the one way a batch goes through the model and its loss.

    cfg = read_config("<run>/config.json")                        # ChainConfig: the fields of a run's config that the data, the network and the loss read
    model, loss, cached = build(cfg)                              # the model and the loss, head layouts matched
    step = load_weights(model, loss, "<run>/ckpt_00001500.pt")    # the weights a run is evaluated with (the EMA average), eval mode
    res, batch = forward(model, batch, dts, device, cfg)          # one batch through the model: ego gaps, device, autocast, the state of the previous key frame
    out, res, gts = forward_loss(model, loss, batch, step_frac, device, cfg)     # ... and its loss with the terms only a training step has
    (net/loss/train_terms.py)

The trainer's config (training/train.py TrainConfig) is ChainConfig plus the engine's fields and the trainer's own. This module stands above the three layers of
net/ (prep < model < loss) and below training/ and evaluation/."""
import dataclasses
import json
from dataclasses import dataclass, field

import torch

from bev_fusion.net.loss.loss_extras import group_gradient_cosines
from bev_fusion.net.loss.losses import BEVLoss, LossConfig, sample_targets
from bev_fusion.net.loss.train_terms import add as add_train_terms
from bev_fusion.net.model.cached_model import CachedBEVFusionModel
from bev_fusion.net.model.interface import check_inputs, check_outputs
from bev_fusion.net.model.model import CAMERAS, BEVFusionModel, ModelConfig
from bev_fusion.net.prep import splits
from bev_fusion.net.prep.batches import batch_ego, to_device

AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


@dataclass
class ChainConfig:
    data_root: str = "~/data/nuscenes_full"
    version: str = "v1.0-trainval"
    train_split: str = "train"
    val_split: str = "val"
    train_scenes: tuple = ()                     # scene names (restrict; empty = the whole split)
    val_scenes: tuple = ()
    feature_cache: str = ""                      # path of the frozen-head feature cache ("" = run the camera heads)
    gt_cache: str = "data/cache/gt"              # GT cache folder of the dataset ("" = off)
    feature_cache2: str = ""                     # second feature cache (BEVFusion, camera_features "dual"; "" = off)
    radar_cache: str = ""                        # per-sample radar cache folder (feature_cache.py; "" = off; exact)
    # s of radar sweeps before the camera time that make one radar set (nuscenes_dataset); 0.09 = the newest sweep of each radar
    radar_window: float = 0.5
    lane_images: str = ""                        # key-frame image cache of the lane branch (net/prep/image_cache.py; "" = off)
    n_cam: int = 3
    cam_stride: int = 2
    # keyframes per clip: > 1 trains on clips of consecutive keyframes of one scene with the world memory carried from keyframe to keyframe (a micro-batch has
    # batch_size * seq_len samples); 1 = independent. The evaluation warms the memory up over seq_len - 1 frames
    seq_len: int = 1
    # decode dt = +-0.5 s with each sample's true keyframe gap and gap-mean ego motion (0.2-0.35 m of label error otherwise)
    gap_ego: bool = True
    # the autocast of the forward: "off" | "bf16" | "fp16" (CUDA only); the engine's field of the same name (trainkit.config)
    amp: str = "off"
    # non-blocking copies of a batch to the GPU (and, in the trainer, a DataLoader with pinned host memory); cuda only, same numbers
    pin_memory: bool = False
    balancing: str = "kendall"                   # task groups: "kendall" (learned weights) | "fixed" (loss.task_weighting)
    consistency: float = 0.0                     # weight of the train-only cross-time consistency (train_terms.py), 0 = off
    consistency_warmup: tuple = (0.2, 0.2)       # (hold, ramp) fractions: off for the first 20 %, then linear in
    # radar_kind "delta_slots": weight of the radar encoder's reconstruction SIDE loss at the start of the training (radar_loss.aux_weight: floor + (start - floor)
    # * 0.5 ** (progress / half)); it helps the first phase, the task losses are the goal; 0 = off (kept-fraction not computed either)
    radar_aux: float = 0.0
    radar_aux_floor: float = 0.05                # weight it decays to
    radar_aux_half: float = 0.2                  # half-life in fractions of the training
    # model.camera_student trainable: weight at the start of the training of the anchor, the distillation loss (student_loss.py) between the student's
    # current-frame outputs and the cached teachers'; keeps the depth from drifting (no depth ground truth); 0 = off (nothing computed, nothing logged)
    student_anchor: float = 0.0
    student_anchor_floor: float = 0.1            # the fraction of student_anchor it falls to ...
    student_anchor_until: float = 0.5            # ... linearly until this fraction of the steps, constant after (student_loss.anchor_weight)
    model: dict = field(default_factory=dict)    # ModelConfig fields
    loss: dict = field(default_factory=dict)     # LossConfig fields


# Options that left the code (they are in the git tag v1): where ("" = the top level of a config, else its dict of that name) -> {key: its off value}. A stored
# config that names one at its off value still loads (it had no effect), at any other value it is refused. A removed KIND of a slot that is still there
# (radar_kind 'pillars', lane_branch 'rgb', balancing 'uwso') is refused where the kinds are checked (model.check_config, build).
REMOVED = {"": {"uwso_T": 5.0},
           "model": {"trunk_kind": "dualbev", "cam_collapse": "zconv", "cam_adapter": False, "fusion": "concat", "share_mover": False, "splat_fused": False,
                     "sync_free": False, "dense_outputs": False, "rad_static_gate": False, "rad_static_vr": 1.0, "refine": False, "refine_k": 50,
                     "refine_jitter": 1, "lane_use_depth": True, "trunk_fpn_original": False},
           "loss": {"sync_free": False, "line_bce": 0.0, "line_bce_pos": 3.0, "line_bce_ignore": 1.0, "consistency": False, "w_consistency": 0.1, "w_refine": 1.0}}


def _switched_on(where, part):
    """the REMOVED options the dict `part` names at another value than off, as texts"""
    return [f"{where + '.' if where else ''}{key} = {part[key]!r}" for key, off in REMOVED[where].items() if key in part and part[key] != off]


def _refuse(bad):
    if bad:
        raise ValueError("the config switches on options that were removed from the code (they are in the git tag v1): " + ", ".join(bad))


def live(where, part):
    """the dict `part` of a config ("model" or "loss") without the REMOVED options it names at their off value: the fields the code has; one named at another
    value is a ValueError"""
    _refuse(_switched_on(where, part))
    return {k: v for k, v in part.items() if k not in REMOVED[where]}


def drop_removed(raw):
    """a config as read from json, for a config class: one REMOVED option at another value than off is a ValueError that lists them all; the removed keys of
    the top level are dropped (the classes do not know them). Those of `model` and `loss` stay in their dicts as the file names them, because a checkpoint
    records these dicts and a resume compares them whole (trainkit.checkpoint.check_resume); `build` leaves them out (`live`)."""
    _refuse([b for where in REMOVED for b in _switched_on(where, (raw.get(where) or {}) if where else raw)])
    return {k: v for k, v in raw.items() if k not in REMOVED[""]}


def read_raw(path):
    """a config file as the config classes take it: the scene lists of its "split" filled in (net/prep/splits.py resolve; a config that carries its lists is
    left as it is), the options that left the code dropped or refused (drop_removed)"""
    with open(path, encoding="utf-8") as fh:
        return drop_removed(splits.resolve(json.load(fh)))


def read_config(path):
    """the ChainConfig of a stored run (its config.json): the keys ChainConfig names; the engine's and the trainer's own keys (steps, lr, workers, ...) and the
    "_" comments are left out, the trainer has checked them (training/train.py read_config refuses an unknown key)"""
    raw = read_raw(path)
    names = {f.name for f in dataclasses.fields(ChainConfig)}
    return ChainConfig(**{k: v for k, v in raw.items() if k in names})


# ------------------------------------------------------------------------------------------------ model and loss
def build(cfg):
    """(model, loss, cached?). The head layout of the model follows the loss's layer spec: the flags that define it (centrelines, dense_objects,
    radar_stationary, centreline_1m) are taken from the LossConfig, so the two cannot disagree; the model returns predictions in the loss's layout
    (res["out"] = {dt: {"sem", "obj", "mass", ...}})."""
    lc = LossConfig(**{**live("loss", cfg.loss), **({"task_weighting": "fixed"} if cfg.balancing == "fixed" else {})})
    loss = BEVLoss(lc)
    if cfg.balancing not in ("kendall", "fixed"):
        raise ValueError(f"balancing {cfg.balancing!r}: kendall or fixed" + (" (uwso was removed in v2; tag v1)" if cfg.balancing == "uwso" else ""))
    layout = {"centrelines": lc.centrelines, "dense_objects": lc.dense_objects, "radar_stationary": lc.radar_stationary, "centreline_1m": lc.centreline_1m}
    mcfg = ModelConfig(**{**layout, **live("model", cfg.model)})
    for k, v in layout.items():
        if getattr(mcfg, k) != v:
            raise ValueError(f"model.{k} = {getattr(mcfg, k)} but loss.{k} = {v}: the head layout must match the loss")
    cached = bool(cfg.feature_cache)
    if mcfg.camera_features != "dualbev" and not cached:
        raise ValueError(f"camera_features={mcfg.camera_features!r} needs a feature_cache")
    if (("feature_cache2" in {s.cache for s in CAMERAS[mcfg.camera_features]}) != bool(cfg.feature_cache2)):
        raise ValueError(f"camera_features {mcfg.camera_features!r} and feature_cache2 do not match: the kinds with a second feature source "
                         "('dual') need feature_cache2 (the BEVFusion cache), and feature_cache2 needs such a kind")
    if mcfg.lane_branch and cached and not cfg.lane_images:
        raise ValueError("model.lane_branch with a feature_cache needs lane_images (build_image_cache.py build-cache)")
    problems = student_findings(cfg, mcfg)
    if problems:
        raise ValueError("; ".join(problems))
    model = CachedBEVFusionModel(mcfg) if cached else BEVFusionModel(mcfg)
    return model, loss, cached


def student_findings(cfg, mcfg):
    """what model.camera_student and its anchor need of the config, as messages (build raises them): the student replaces cached outputs of the current frame
    (student_frames 'all': of every frame of the window), so a feature cache and the key-frame image folder (lane_images); the anchor needs a student that
    trains, and a schedule that makes sense"""
    out = []
    if mcfg.camera_student and not cfg.feature_cache:
        out.append("model.camera_student replaces cached teacher outputs: it needs a feature_cache")
    if mcfg.camera_student and not cfg.lane_images:
        out.append("model.camera_student reads the current key-frame image: it needs lane_images (the folder of the key-frame JPEGs, data/cache/img_front)")
    if mcfg.camera_student and mcfg.student_frames == "all" and cfg.n_cam < 2:
        out.append("model.student_frames 'all' feeds the student the older frames of the window: it needs n_cam >= 2")
    if cfg.student_anchor < 0 or not 0 <= cfg.student_anchor_floor <= 1 or not 0 < cfg.student_anchor_until <= 1:
        out.append(f"student_anchor {cfg.student_anchor} (>= 0), student_anchor_floor {cfg.student_anchor_floor} (0..1), student_anchor_until {cfg.student_anchor_until} (0 < until <= 1)")
    if cfg.student_anchor > 0 and not (mcfg.camera_student and mcfg.student_trainable):
        out.append("student_anchor > 0 pulls a trainable student towards the cached teachers: it needs model.camera_student and model.student_trainable")
    return out


# ------------------------------------------------------------------------------------------------ weights
def external(model):
    """the state prefixes that are not in a checkpoint: the frozen camera heads (they come from the pre-trained file) and a frozen camera student
    (model.student_trainable false: it is loaded from student_init). With freeze_camera false the camera weights are trained, so they belong to the state: a
    resume must restore them, not start them
    from the pre-trained file again; a student that trains is part of the state (checkpoints, resume, EMA)"""
    cfg = model.cfg
    return (("camera.",) if cfg.freeze_camera else ()) + (("student.",) if cfg.camera_student and not cfg.student_trainable else ())


def load_weights(model, loss, path):
    """the weights a run is evaluated with, from its checkpoint `path` (the trainer's format: trainkit.checkpoint): the trained state of the model and the loss,
    then the EMA average over the model's (what the trainer validates under); the model is left in eval mode. A model key that the checkpoint lacks and that is
    not `external`, or a key
    the model does not have, is a RuntimeError. -> the checkpoint's step"""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck["state"]
    found = model.load_state_dict({k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}, strict=False)
    missing = [k for k in found.missing_keys if not k.startswith(external(model))]
    if missing or found.unexpected_keys:
        raise RuntimeError(f"{path} does not fit this model: {len(missing)} keys it lacks (e.g. {missing[:2]}), {len(found.unexpected_keys)} keys the model does not have "
                           f"(e.g. {found.unexpected_keys[:2]}): a checkpoint of another code version or config")
    loss.load_state_dict({k[len("loss."):]: v for k, v in state.items() if k.startswith("loss.")})
    extra = model.load_state_dict(ck["ema"], strict=False).unexpected_keys
    if extra:
        raise RuntimeError(f"{path}: the EMA holds {len(extra)} keys the model does not have (e.g. {extra[:2]})")
    model.eval()
    return ck["step"]


# ------------------------------------------------------------------------------------------------ forward
def autocast(amp, device):
    """the autocast context of the forward pass: bf16 / fp16 on CUDA, nothing otherwise; an unknown name is an error (as the engine's
    trainkit.amp.amp_context)"""
    if amp != "off" and amp not in AMP_DTYPES:
        raise ValueError(f"amp must be one of off, {', '.join(AMP_DTYPES)}; got {amp!r}")
    if amp in AMP_DTYPES and str(device).startswith("cuda"):
        return torch.autocast("cuda", dtype=AMP_DTYPES[amp])
    return torch.autocast("cpu", enabled=False)


_CONTRACT_CHECKED = []      # interface.check_inputs / check_outputs run on the first forward of the process


def forward(model, batch, dts, device, cfg, strict=False, state=None, step=None):
    """one batch (still on the host) through the model at the times `dts` -> (the model's result, the batch on the device). With cfg.gap_ego the model decodes
    dt = +-0.5 s with the TRUE gap of each sample and its gap-mean ego motion (batch["gap_steps"], made from the poses: no ground truth is read); strict: a dt
    without such a gap is an
    error (LossConfig.hygiene_2026_11_02). state / step: the model state of the previous keyframe and the ego step since then (batches.clip_step). The interface
    contract is checked on the first batch of the process (training, validation and the evaluation all pass here)."""
    ego = batch_ego(batch["gap_steps"], dts, device, strict=strict) if cfg.gap_ego else None
    batch = to_device(batch, device, non_blocking=cfg.pin_memory)
    kw = {k: v for k, v in (("ego", ego), ("state", state), ("step", step)) if v is not None}
    first = not _CONTRACT_CHECKED
    if first:
        check_inputs(batch, dts=dts, ego=ego, raise_on=True)
    with autocast(cfg.amp, device):
        res = model(batch, dts=dts, **kw)
    if first:
        check_outputs(res, dts, raise_on=True)
        _CONTRACT_CHECKED.append(True)
    return res, batch


def forward_loss(model, loss, batch, step_frac, device, cfg, diag=False, train_terms=True, state=None, step=None):
    """-> (loss dict, model result, GT targets) of one batch: `forward` at the target times of the GT (0 and +-0.5 s), BEVLoss on the predictions cast to
    float32, then the train-only terms (train_terms.py; train_terms False = validation: the total stays the task loss). With diag the cosines of the task-group
    gradients at the BEV feature
    are put into out['diag']."""
    gts = sample_targets(batch["gt"])
    dts = tuple(sorted(gts))
    res, batch = forward(model, batch, dts, device, cfg, strict=loss.cfg.hygiene_2026_11_02, state=state, step=step)
    with torch.autocast("cpu", enabled=False), torch.autocast("cuda", enabled=False):
        preds = {dt: {k: ({a: b.float() for a, b in v.items()} if isinstance(v, dict) else v.float()) for k, v in p.items()}
                 for dt, p in res["out"].items()}                                    # already in the loss's layout
        out = loss(preds, gts, step_frac)
        out["total_main"] = out["total"].detach()                                 # without the train-only terms
        add_train_terms(out, model, loss, res, gts, batch, cfg, step_frac, device, train_terms)
        if diag:
            groups = {k[len("group."):]: v for k, v in out.items() if k.startswith("group.") and torch.is_tensor(v) and v.requires_grad}
            if "consistency" in out and out["consistency"].requires_grad:      # its gradient norm at the BEV feature calibrates its weight
                groups["consistency"] = out["consistency"]
            if len(groups) > 1 and res["features"].requires_grad:      # a frozen trunk gives the groups no common feature to differentiate by
                names, C, nrm = group_gradient_cosines(groups, res["features"])
                out["diag"] = {"groups": names, "cos": C.tolist(), "norm": nrm.tolist()}
    return out, res, gts
