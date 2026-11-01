"""The two interfaces of the network (layers 2 and 4 of training/architecture.md): what the network reads (BEVInputs)
and what it returns (BEVOutputs), written down once with names, shapes, units and frames, and checked.

    check_inputs(batch, dts=..., state=..., step=..., ego=...)    -> list of problems (empty = fine); raise_on=True raises
    check_outputs(res, dts)                                        -> the same for the result of BEVFusionModel.forward

Both stay plain dicts (the layout of nuscenes_dataset.collate / feature_cache.collate_cached and of BEVFusionModel.forward), so the
trainer, the evaluation and every stored checkpoint keep working; the tables below are the contract a slot can rely on. Frames:
the vehicle frame of the current keyframe (x forward, y left, metres); B samples, F camera frames (0 = now, then older), X x Y grid.
Run the self-test: python training/interface.py
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
    "img": (False, "(B, F, 3, H, W) images for the live camera heads; the cached chain carries a (B, F, 1) placeholder and reads prob / ctx"),
    "cam_vec": (False, "(B, F, 27) camera vector of the DualBEV depth net; live camera heads only"),
    "prob": (False, "(B, F, D, h, w) cached depth probabilities of the frozen image heads (camera_features dualbev / dual)"),
    "ctx": (False, "(B, F, 80, h, w) cached context features of the frozen image heads"),
    "hw": (False, "(H, W) image size the cached features belong to"),
    "prob2": (False, "(B, F, D2, h2, w2) cached BEVFusion depth (camera_features dual)"),
    "ctx2": (False, "(B, F, 80, h2, w2) cached BEVFusion context (camera_features dual)"),
    "ida2": (False, "crop / resize of the BEVFusion features (camera_features dual)"),
    "lane_img": (False, "(B, 3, H, W) uint8 image of the current frame for the lane add-on"),
    "flip": (False, "(B,) bool: samples mirrored by the augmentation (y -> -y); the camera BEV is mirrored after the camera slot"),
    "radar_poses": (False, "(B, 5, 4, 4) sensor -> vehicle matrices of the five radars (the rig calibration; precompute.py makes vis_radar from it)"),
    "vis_cam": (False, "(B, 1, X, Y) cells the current camera frame observes (precompute.py; made from K, T, crop and grid when absent)"),
    "vis_radar": (False, "(B, 1, X, Y) cells the radar rig can see, the fans (precompute.py; D36)"),
    "vis_fusion": (False, "(B, 1, X, Y) the fusion area = vis_cam x vis_radar (precompute.py; D37: fusion results are scored here)"),
    "radar_hits": (False, "(B, 1, X, Y) cells with radar detections of this frame (precompute.py): the radar memory's write mask"),
    "gt": (False, "list (B) of GT dicts; read inside the network only by the refiner option (training)"),
    "tokens": (False, "list (B) of sample tokens (bookkeeping)"),
    "steps_raw": (False, "per sample the steps as python floats (bookkeeping)"),
}
# the fields of every radar detection in batch["radar"] (all (N,) tensors; nuscenes_dataset.FrontRadarDataset._radar)
RADAR_FIELDS = {"x": "m, vehicle frame of now", "y": "m", "b": "sample index in the batch (long)", "r": "range from the sensor, m",
                "psi": "ray direction in the vehicle frame, rad", "vr": "radial velocity (Doppler, ego-compensated), m/s",
                "rcs": "radar cross section, dBsm", "dt": "age of the sweep, s"}
# only frame 0 of K and T_cam_car is read today (the lift and the visibility of the current frame); the per-frame entries are kept
CALL = {
    "dts": "target times (s, either sign) the heads decode; the output keys are these nominal dt",
    "state": "per-branch temporal states of the previous call ({'cam': ..., 'rad': ...}) or None",
    "step": "the ego step (dt, v, omega) since the previous call, scalars or (B,) tensors, or None",
    "ego": "{nominal dt: (dt, v, omega)} as (B,) tensors: the true gap to the target keyframe and the gap-mean ego motion, or None",
}
# per target time dt: name -> meaning; plus the call-level keys
OUTPUTS_PER_DT = {
    "sem": "(B, L, X, Y) map head: layer_spec channels (area / line logits, distances in metres, ...)",
    "obj": "dict of object head maps: hm (B, G, X, Y) logits, off, size, rot, rad, vel (+ dir, mov)",
    "mass": "(B, 1, X, Y) splat mass of the time mover at dt",
}
OPTIONAL_PER_DT = {"boxes": "dense box output (extract.dense_boxes): peak weight w (B, G, X, Y) and the box parameters per cell",
                   "lines": "soft line map (B, n_lines, X, Y) from the distance channels (extract.soft_line_map)",
                   "v_hat": "velocity field at dt = 0", "cam_aux": "camera aux head (training)", "radar_aux": "radar aux head (training)",
                   "obj_feat": "refiner features", "obj_dt": "refiner dt", "refine": "refiner training targets"}
OPTIONAL_OUTPUTS = {"radar_side": "radar_kind 'delta_slots': {'features': (B, C, X, Y) encoder output, 'state': (B, 61, X, Y) lossless slot raster}, "
                                  "read by the radar side loss (train.py) and the kept-fraction monitor"}
OUTPUTS = {"out": "{dt: per-dt dict}", "state": "{'cam': ..., 'rad': ...} temporal states for the next call",
           "features": "(B, C, X, Y) BEV features S after the backbone (and neck)"}


def _batch_size(batch):
    for k in ("K", "img", "prob"):
        if k in batch and torch.is_tensor(batch[k]):
            return batch[k].shape[0], batch[k].shape[1]
    return None, None


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
        if "radar_poses" in batch and tuple(batch["radar_poses"].shape[1:]) != (5, 4, 4):
            errs.append(f"radar_poses: shape {tuple(batch['radar_poses'].shape)} != (B, 5, 4, 4)")
        for k in ("vis_cam", "vis_radar", "vis_fusion", "radar_hits"):
            if k in batch and tuple(batch[k].shape[:2]) != (B, 1):
                errs.append(f"{k}: leading shape {tuple(batch[k].shape[:2])} != (B, 1) = {(B, 1)}")
        if "steps" in batch and len(batch["steps"]) != Fn:
            errs.append(f"steps: {len(batch['steps'])} frames != F = {Fn}")
        rad = batch.get("radar", {})
        if "b" in rad and len(rad["b"]) and int(rad["b"].max()) >= B:
            errs.append(f"radar: sample index {int(rad['b'].max())} >= B = {B}")
        if rad:
            errs += [f"radar: missing field {f!r} ({m})" for f, m in RADAR_FIELDS.items() if f not in rad]
            n = {len(v) for v in rad.values() if torch.is_tensor(v)}
            if len(n) > 1:
                errs.append(f"radar: fields of different lengths {sorted(n)}")
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
        for k, v in list(p.items()) + list((p.get("obj") or {}).items()) + list((p.get("boxes") or {}).items()):
            if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all():
                errs.append(f"dt {dt}: {k} not finite")
    if raise_on and errs:
        raise ValueError("BEVOutputs: " + "; ".join(errs))
    return errs


def _selftest():
    B, Fn = 2, 3
    z = torch.zeros
    radar = dict({f: z(5) for f in RADAR_FIELDS}, b=torch.tensor([0, 0, 1, 1, 1]))
    batch = {"K": z(B, Fn, 3, 3), "T_cam_car": z(B, Fn, 4, 4), "ida": None, "v": z(B), "omega": z(B),
             "steps": [[] for _ in range(Fn)], "radar": radar,
             "prob": z(B, Fn, 4, 2, 3), "ctx": z(B, Fn, 80, 2, 3), "hw": (16, 24), "tokens": ["a", "b"]}
    assert check_inputs(batch) == [], check_inputs(batch)
    e = check_inputs(dict(batch, radar={k: v for k, v in radar.items() if k != "vr"}))
    assert any("'vr'" in x for x in e), e
    assert any("lengths" in x for x in check_inputs(dict(batch, radar=dict(radar, rcs=z(4)))))
    bad = dict(batch, v=z(B + 1), extra=1)
    del bad["omega"]
    e = check_inputs(bad)
    assert any("omega" in x for x in e) and any("extra" in x for x in e) and any("v:" in x for x in e), e
    bad = dict(batch, ctx=torch.full((B, Fn, 80, 2, 3), float("nan")))
    assert any("not finite" in x for x in check_inputs(bad))
    try:
        check_inputs(dict(batch, radar=dict({f: z(1) for f in RADAR_FIELDS}, b=torch.tensor([3]))), raise_on=True)
        raise AssertionError("no raise")
    except ValueError as ex:
        assert "sample index" in str(ex)
    p = {"sem": z(B, 12, 8, 6), "obj": {"hm": z(B, 3, 8, 6)}, "mass": z(B, 1, 8, 6)}
    res = {"out": {0.0: dict(p, v_hat=z(B, 2, 8, 6)), 0.5: p}, "state": {}, "features": z(B, 4, 8, 6)}
    assert check_outputs(res, (0.0, 0.5)) == [], check_outputs(res, (0.0, 0.5))
    assert any("times" in x for x in check_outputs(res, (0.0,)))
    res2 = {"out": {0.0: {"sem": z(B, 12, 8, 6), "obj": {}, "foo": 1}}, "state": {}, "features": None}
    e = check_outputs(res2, (0.0,))
    assert any("mass" in x for x in e) and any("foo" in x for x in e) and any("v_hat" in x for x in e), e
    bad_boxes = {"out": {0.5: dict(p, boxes={"w": torch.full((B, 3, 8, 6), float("inf"))})}, "state": {}, "features": None}
    assert any("w not finite" in x for x in check_outputs(bad_boxes, (0.5,))), "the dense boxes are checked"
    print("interface selftest passed: input and output contracts checked (missing, unknown, shape, finite)")


if __name__ == "__main__":
    _selftest()
