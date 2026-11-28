"""The two interfaces of the network (layers 2 and 4 of training/architecture.md): what the network reads (BEVInputs)
and what it returns (BEVOutputs), written down once with names, shapes, units and frames, and checked.

    check_inputs(batch, dts=..., state=..., step=..., ego=...)    -> list of problems (empty = fine); raise_on=True raises
    check_outputs(res, dts)                                        -> the same for the result of BEVFusionModel.forward

Both stay plain dicts (the layout of nuscenes_dataset.collate / feature_cache.collate_cached and of BEVFusionModel.forward), so the
trainer, the evaluation and every stored checkpoint keep working; the tables below are the contract a slot can rely on. Frames:
the vehicle frame of the current keyframe (x forward, y left, metres); B samples, F camera frames (0 = now, then older), X x Y grid.
Tests: tests/test_interface.py
"""
import torch

# name: (required, meaning) - the network reads nothing else
INPUTS = {
    "K": (True, "(B, F, 3, 3) camera intrinsics per frame, pixels of the resized and cropped image"),
    "T_cam_car": (True, "(B, F, 4, 4) camera pose in the vehicle frame of its own time"),
    "ida": (True, "image crop / resize of the batch (one per batch: items[0]); with the cached features the crop they were made with"),
    "steps": (True, "per frame k a list of ego steps (dt, v, omega), each three (B,) tensors, from frame k to now; padded with zero steps"),
    "v": (True, "(B,) ego speed at the keyframe, m/s"),
    "omega": (True, "(B,) ego yaw rate at the keyframe, rad/s"),
    "radar": (True, "dict of (N,) tensors: detections of 0.5 s of sweeps of all radars compensated to now, with 'b' = the sample index"),
    "radar_seq": (False, "radar_per_frame (t25): list of F radar dicts as 'radar', one per camera frame at THAT frame's camera time in that frame's ego frame (0 = now = 'radar'); "
                         "read with ModelConfig.radar_per_frame, otherwise ignored"),
    "img": (False, "(B, F, 3, H, W) images for the live camera heads; the cached chain carries a (B, F, 1) placeholder and reads prob / ctx"),
    "cam_vec": (False, "(B, F, 27) camera vector of the DualBEV depth net; live camera heads only"),
    "prob": (False, "(B, F, D, h, w) cached depth probabilities of the frozen image heads (camera_features dualbev / dual)"),
    "ctx": (False, "(B, F, 80, h, w) cached context features of the frozen image heads"),
    "hw": (False, "(H, W) image size the cached features belong to"),
    "prob2": (False, "(B, F, D2, h2, w2) cached BEVFusion depth (camera_features dual)"),
    "ctx2": (False, "(B, F, 80, h2, w2) cached BEVFusion context (camera_features dual)"),
    "ida2": (False, "crop / resize of the BEVFusion features (camera_features dual)"),
    "lane_img": (False, "(B, 3, H, W) uint8 image of the current frame for the lane add-on; it also feeds the camera student (ModelConfig.camera_student, t22), which REQUIRES it: "
                        "the contract cannot see the config, so the cached model raises a KeyError naming TrainConfig.lane_images when it is missing"),
    "lane_img_old": (False, "(B, F-1, 3, H, W) uint8 images of the OLDER frames of the window (frames 1 .. F-1, <lane_images>/sweeps); only with ModelConfig.student_frames 'all' (t26), which REQUIRES it for "
                            "F > 1: the cached model raises a KeyError naming the sweeps cache when it is missing"),
    "img_jitter": (False, "dict from augment.draw_jitter (u (4, B), strength, noise_seed): the photometric jitter of the camera student's input of this call (TrainConfig.img_jitter, training only; "
                          "the cached model applies it in training mode and ignores it otherwise)"),
    "flip": (False, "(B,) bool: samples mirrored by the augmentation (y -> -y); the camera BEV is mirrored after the camera slot"),
    "gap_steps": (False, "list (B) of {'prev': steps, 'next': steps}: the ego steps (dt, v, omega) from now to the neighbouring keyframes "
                         "(python floats, from the poses; dt < 0 backwards): the true gap and ego motion for decoding dt = +-0.5 s and the memory step between calls"),
    "radar_poses": (False, "(B, 5, 4, 4) sensor -> vehicle matrices of the five radars (the rig calibration; precompute.py makes vis_radar from it)"),
    "vis_cam": (False, "(B, 1, X, Y) cells the current camera frame observes (precompute.py; made from K, T, crop and grid when absent)"),
    "vis_radar": (False, "(B, 1, X, Y) cells the radar rig can see, the fans (precompute.py; D36)"),
    "vis_fusion": (False, "(B, 1, X, Y) the fusion area = vis_cam x vis_radar (precompute.py; D37: fusion results are scored here)"),
    "radar_hits": (False, "(B, 1, X, Y) cells with radar detections of this frame (precompute.py): the radar memory's write mask"),
    "gt": (False, "list (B) of GT dicts: the loss reads it, the network never does"),
    "tokens": (False, "list (B) of sample tokens (bookkeeping)"),
    "steps_raw": (False, "per sample the steps as python floats (bookkeeping)"),
}
# the fields of every radar detection in batch["radar"] (all (N,) tensors; nuscenes_dataset.FrontRadarDataset._radar)
RADAR_FIELDS = {"x": "m, vehicle frame of now", "y": "m", "b": "sample index in the batch (long)", "r": "range from the sensor, m",
                "psi": "ray direction in the vehicle frame, rad", "vr": "radial velocity (Doppler, ego-compensated), m/s",
                "rcs": "radar cross section, dBsm", "dt": "age of the sweep, s"}
# only frame 0 of K and T_cam_car is read today (the lift and the visibility of the current frame); the per-frame entries are kept
# per target time dt: name -> meaning; plus the call-level keys
OUTPUTS_PER_DT = {
    "sem": "(B, L, X, Y) map head: layer_spec channels (area / line logits, distances in metres, ...)",
    "obj": "dict of object head maps: hm (B, G, X, Y) logits, off, size, rot, rad, vel (+ dir, mov)",
    "mass": "(B, 1, X, Y) splat mass of the time mover at dt",
}
OPTIONAL_PER_DT = {"v_hat": "velocity field at dt = 0", "v_levels": "motion block (t15): the coarser levels' fields at dt = 0, dict l0, l1, ... (coarse to fine) of (B, 2, X, Y)",
                   "cam_aux": "camera aux head (training)", "radar_aux": "radar aux head (training)",
                   "sem_1m": "centreline_1m option (D48): the 1 m centre-line head, logits (B, C, X/2, Y/2), channels from losses.layer_spec_1m"}
OUTPUTS = {"out": "{dt: per-dt dict}", "state": "{'cam': ..., 'rad': ...} temporal states for the next call",
           "features": "(B, C, X, Y) BEV features S after the backbone (and neck)"}


def _batch_size(batch):
    for k in ("K", "img", "prob"):
        if k in batch and torch.is_tensor(batch[k]):
            return batch[k].shape[0], batch[k].shape[1]
    return None, None


def _radar_problems(name, rad, B):
    """the problems of one radar dict (batch["radar"], or one set of batch["radar_seq"]): sample index, fields, field lengths"""
    errs = []
    if "b" in rad and len(rad["b"]) and int(rad["b"].max()) >= B:
        errs.append(f"{name}: sample index {int(rad['b'].max())} >= B = {B}")
    if rad:
        errs += [f"{name}: missing field {f!r} ({m})" for f, m in RADAR_FIELDS.items() if f not in rad]
        n = {len(v) for v in rad.values() if torch.is_tensor(v)}
        if len(n) > 1:
            errs.append(f"{name}: fields of different lengths {sorted(n)}")
    return errs


def check_inputs(batch, dts=(0.0,), state=None, step=None, ego=None, raise_on=False):
    """the input contract: required keys present, unknown keys named, shapes consistent in B and F, tensors finite"""
    errs = [f"missing input {k!r}: {m}" for k, (req, m) in INPUTS.items() if req and k not in batch]
    errs += [f"unknown input {k!r} (add it to interface.INPUTS with its meaning)" for k in batch if k not in INPUTS]
    B, Fn = _batch_size(batch)
    if B is not None:
        for k in ("K", "T_cam_car", "img", "cam_vec", "prob", "ctx", "prob2", "ctx2"):
            if k in batch and tuple(batch[k].shape[:2]) != (B, Fn):
                errs.append(f"{k}: leading shape {tuple(batch[k].shape[:2])} != (B, F) = {(B, Fn)}")
        for k in ("v", "omega", "flip", "radar_poses"):
            if k in batch and batch[k].shape[0] != B:
                errs.append(f"{k}: {batch[k].shape[0]} samples != B = {B}")
        if "lane_img_old" in batch and tuple(batch["lane_img_old"].shape[:2]) != (B, Fn - 1):       # t26 student_frames 'all': one image per older frame
            errs.append(f"lane_img_old: leading shape {tuple(batch['lane_img_old'].shape[:2])} != (B, F - 1) = {(B, Fn - 1)}")
        if "radar_poses" in batch and tuple(batch["radar_poses"].shape[1:]) != (5, 4, 4):
            errs.append(f"radar_poses: shape {tuple(batch['radar_poses'].shape)} != (B, 5, 4, 4)")
        for k in ("vis_cam", "vis_radar", "vis_fusion", "radar_hits"):
            if k in batch and tuple(batch[k].shape[:2]) != (B, 1):
                errs.append(f"{k}: leading shape {tuple(batch[k].shape[:2])} != (B, 1) = {(B, 1)}")
        if "steps" in batch and len(batch["steps"]) != Fn:
            errs.append(f"steps: {len(batch['steps'])} frames != F = {Fn}")
        errs += _radar_problems("radar", batch.get("radar", {}), B)
        if "radar_seq" in batch:
            if len(batch["radar_seq"]) != Fn:
                errs.append(f"radar_seq: {len(batch['radar_seq'])} radar sets != F = {Fn}")
            for k, rad in enumerate(batch["radar_seq"]):
                errs += _radar_problems(f"radar_seq[{k}]", rad, B)
    for k, v in batch.items():
        if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all():
            errs.append(f"{k}: not finite")
    if ego is not None and set(map(float, ego)) - set(map(float, dts)):
        errs.append(f"ego has times {sorted(set(map(float, ego)) - set(map(float, dts)))} that are not in dts")
    if raise_on and errs:
        raise ValueError("BEVInputs: " + "; ".join(errs))
    return errs


def check_outputs(res, dts, raise_on=False):
    """the output contract: every dt present with its maps, shapes consistent, values finite"""
    errs = [f"missing output {k!r}" for k in OUTPUTS if k not in res]
    out = res.get("out", {})
    if sorted(map(float, out)) != sorted(map(float, dts)):
        errs.append(f"out has times {sorted(out)} != dts {sorted(dts)}")
    shape = None
    for dt, p in out.items():
        errs += [f"dt {dt}: missing {k!r}" for k in OUTPUTS_PER_DT if k not in p]
        if abs(float(dt)) < 1e-6 and "v_hat" not in p:
            errs.append(f"dt {dt}: missing 'v_hat' (the velocity field is made at dt = 0)")
        errs += [f"dt {dt}: unknown output {k!r} (add it to interface.OPTIONAL_PER_DT)" for k in p
                 if k not in OUTPUTS_PER_DT and k not in OPTIONAL_PER_DT]
        if "sem" in p:
            s = (p["sem"].shape[0],) + tuple(p["sem"].shape[-2:])
            shape = shape or s
            if s != shape:
                errs.append(f"dt {dt}: sem (B, X, Y) {s} != {shape}")
        for k, v in list(p.items()) + list((p.get("obj") or {}).items()) + list((p.get("v_levels") or {}).items()):
            if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all():
                errs.append(f"dt {dt}: {k} not finite")
    if raise_on and errs:
        raise ValueError("BEVOutputs: " + "; ".join(errs))
    return errs
