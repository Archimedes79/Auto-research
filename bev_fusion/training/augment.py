"""Mirror augmentation of a training batch: the world is mirrored about the vehicle's x axis (y -> -y), the standard BEV
augmentation of the BEV detectors (BEVDet's BDA, BEVFusion) - the frozen camera features are cached, so the mirror is done in
the BEV (after the lift) and on every other input and target:

  camera BEV and camera visibility   flipped along y inside BEVFusionModel.forward (batch["flip"], bool (B,))
  radar detections                   y -> -y, ray direction psi -> -psi (range, Doppler along the ray, rcs, age unchanged); also every
                                     set of batch["radar_seq"] (radar_per_frame, t25)
  ego motion                         yaw rate omega -> -omega (batch["omega"], batch["steps"], the GT meta steps)
  GT rasters (X, Y)                  flipped along the last axis; velocity (vx, vy) and centreline direction (cos, sin): the
                                     lateral component changes sign
  GT boxes (meta)                    centre y -> -y, yaw -> -yaw, velocity vy -> -vy
Not mirrored (not used in training): meta K, T_c2e, T_e2g, vectors, corners.
The grid is symmetric about y = 0 (y -26..26 m), so cell iy of a mirrored map is cell NY - 1 - iy. A clip (sequence training)
is mirrored as a whole: the same mask for every keyframe, so the carried memory and the ego steps stay consistent.

    flip_batch(batch, flip)   in place on a collated batch (CPU, before to_device); flip: bool (B,)
Photometric jitter of the camera student's image (t26, TrainConfig.img_jitter / img_jitter_from): draw_jitter makes batch["img_jitter"] (one draw per sample, a clip is one sample; from
(seed, step, micro-batch) through jitter_seed, so a resume repeats it), the cached model applies jitter_images to the student's input in training mode only. Tests: tests/test_student_window.py.
Run the self-test (nuScenes mini): python training/augment.py
"""
import math

import numpy as np
import torch

LATERAL_VECTOR_KEYS = ("velocity", "flow_ego", "flow", "centreline_dir")      # (2, X, Y): component 1 is the lateral one


def flip_meta(m):
    out = dict(m)
    boxes = []
    for b in m.get("boxes", []):
        b = dict(b)
        c = list(b["center"])
        c[1] = -c[1]
        b["center"] = c
        b["yaw"] = -b["yaw"]
        v = list(b["vel"])
        v[1] = -v[1]
        b["vel"] = v
        if "corners" in b:
            b["corners"] = [[x, -y] for x, y in b["corners"]]
        boxes.append(b)
    out["boxes"] = boxes
    if "steps" in m:
        out["steps"] = [(dt, v, -w) for dt, v, w in m["steps"]]
    return out


def flip_gt(g):
    """One GT dict of gt_bev (arrays [.., ix, iy] and meta) -> its mirror image (new arrays; the input is not changed)."""
    out = {}
    for k, v in g.items():
        if k == "meta":
            out[k] = flip_meta(v)
            continue
        a = np.flip(np.asarray(v), axis=-1).copy()
        if k in LATERAL_VECTOR_KEYS:
            a[1] = -a[1]
        out[k] = a
    return out


def flip_radar(rad, flip):
    """a radar dict (nuscenes_dataset collate) with y and psi negated where flip[b] (bool (B,), on the host); a new dict, the input is not changed"""
    rad = dict(rad)
    f = flip[rad["b"].cpu()].to(rad["y"].device)
    rad["y"] = torch.where(f, -rad["y"], rad["y"])
    rad["psi"] = torch.where(f, -rad["psi"], rad["psi"])
    return rad


def flip_batch(batch, flip):
    """Mirror the samples b with flip[b] of a collated batch, in place; sets batch["flip"] (bool (B,)) for the model."""
    flip = torch.as_tensor(flip, dtype=torch.bool).cpu()
    batch["flip"] = flip
    if not bool(flip.any()):
        return batch
    sign = torch.where(flip, -1.0, 1.0)
    batch["omega"] = batch["omega"] * sign.to(batch["omega"].dtype)
    batch["steps"] = [[(dt, v, w * sign.to(w.dtype)) for dt, v, w in fr] for fr in batch["steps"]]
    if "gap_steps" in batch:
        batch["gap_steps"] = [{k: [(dt, v, -w) for dt, v, w in s] for k, s in g.items()} if flip[i] else g for i, g in enumerate(batch["gap_steps"])]
    batch["radar"] = flip_radar(batch["radar"], flip)
    if "radar_seq" in batch:                                     # radar_per_frame (t25): the radar set of every camera frame
        batch["radar_seq"] = [flip_radar(r, flip) for r in batch["radar_seq"]]
    if "gt" in batch:
        batch["gt"] = [{key: flip_gt(g) for key, g in s.items()} if flip[i] else s for i, s in enumerate(batch["gt"])]
    for key in ("vis_cam", "vis_radar", "vis_fusion", "radar_hits", "radar_static"):    # precomputed maps (precompute.MAPS) already in the batch
        if key in batch:
            m = batch[key]
            batch[key] = torch.where(flip.to(m.device).view(-1, 1, 1, 1), m.flip(-1), m)
    return batch


# ---- photometric jitter of the camera student's input (t26, TrainConfig.img_jitter): the lift reads cached tables of the camera geometry, so only the colours change here
JITTER_RANGE = 0.4          # brightness, contrast and saturation factors: 1 +- 0.4 x strength
JITTER_GAMMA = 1.25         # gamma between 1 / 1.25 and 1.25 at strength 1 (an exponent 1.25 ** (strength u), u in [-1, 1])
JITTER_NOISE = 0.02         # sigma of the gaussian noise in the 0 .. 1 range of the image, x strength
LUMA = (0.299, 0.587, 0.114)


def jitter_seed(seed, step, micro):
    """one integer for (run seed, optimiser step, micro-batch of it): the draw of a step is a function of these three, not of a random stream, so a resume of the pass repeats it"""
    return (int(seed) * 1_000_003 + int(step)) * 1_009 + int(micro)


def draw_jitter(B, strength, seed, key=0):
    """the photometric draw of B samples (batch["img_jitter"]): u (4, B) in [-1, 1] = brightness, contrast, saturation, gamma, from `seed` alone (the same for every `key`, so the key frames of a clip
    see one scene), and the seed of the pixel noise, which differs per `key` (the key frame of the clip). On the CPU."""
    g = torch.Generator().manual_seed(int(seed))
    return {"u": torch.rand(4, B, generator=g) * 2 - 1, "strength": float(strength), "noise_seed": int(seed) * 64 + int(key)}


def jitter_images(img, j, salt=0):
    """uint8 images (B, 3, H, W) or (B, K, 3, H, W) -> float32 0 .. 255 of the same shape, on the device of `img`: brightness, contrast (about the mean luma of the frame), saturation, gamma and gaussian
    noise, in this order, with the factors of sample b from j["u"][:, b] for EVERY frame of the sample (one consistent scene) and the noise drawn per pixel from a generator seeded with
    (j["noise_seed"], salt): the current frame and the older ones take different salts. Deterministic, no state shared with torch's global generator."""
    s, u = j["strength"], j["u"].to(img.device)
    lead = (img.shape[0],) + (1,) * (img.dim() - 1)
    f = lambda a: (1 + JITTER_RANGE * s * u[a]).view(lead)                        # noqa: E731
    luma = lambda x: (x * x.new_tensor(LUMA).view(3, 1, 1)).sum(-3, keepdim=True)  # noqa: E731
    x = (img.float() / 255 * f(0)).clamp(0, 1)
    m = luma(x).mean((-3, -2, -1), keepdim=True)
    x = ((x - m) * f(1) + m).clamp(0, 1)
    y = luma(x)
    x = ((x - y) * f(2) + y).clamp(0, 1)
    x = x ** (JITTER_GAMMA ** (s * u[3])).view(lead)
    g = torch.Generator(device=img.device).manual_seed(j["noise_seed"] * 2 + salt)
    x = x + JITTER_NOISE * s * torch.randn(x.shape, generator=g, device=x.device)
    return x.clamp(0, 1) * 255


def flip_cam(cam_frames, cam_obs, flip):
    """The model's side: (B, F, C, X, Y) camera BEVs and (B, 1, X, Y) visibility (or None) mirrored where flip (B,)."""
    fl = flip.to(cam_frames.device)
    cam_frames = torch.where(fl.view(-1, 1, 1, 1, 1), cam_frames.flip(-1), cam_frames)
    if cam_obs is not None:
        cam_obs = torch.where(fl.view(-1, 1, 1, 1), cam_obs.flip(-1), cam_obs)
    return cam_frames, cam_obs


# ----------------------------------------------------------------------------------------------- self-test
def _principal_axis(xs, ys):
    c = np.cov(np.stack([xs, ys]))
    w, v = np.linalg.eigh(c)
    return math.atan2(v[1, 1], v[0, 1]), w[1] / max(w[0], 1e-9)


def _selftest():
    import copy
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from losses import BEVLoss, sample_targets
    from model import BEVFusionModel, ModelConfig
    from nuscenes_dataset import FrontRadarDataset, collate
    from radar_bev import rasterize
    from temporal_fusion import advance_pose
    from train import gap_ego_motion
    torch.manual_seed(0)
    ds = FrontRadarDataset(os.path.expanduser("~/data/nuscenes"), "v1.0-mini", n_cam=3, cam_stride=2, with_gt=True)
    batch = collate([ds[10], ds[200]])
    orig = copy.deepcopy(batch)

    def eq(x, y):
        if isinstance(x, dict):
            return set(x) == set(y) and all(eq(x[k], y[k]) for k in x)
        if isinstance(x, (list, tuple)) and not isinstance(x, np.ndarray):
            return len(x) == len(y) and all(eq(p, q) for p, q in zip(x, y))
        if isinstance(x, (np.ndarray, float, int)) and not isinstance(x, bool):     # equal_nan: boxes of unknown velocity carry
            return bool(np.allclose(np.asarray(x, float), np.asarray(y, float), atol=1e-9, equal_nan=True))   # vel_sigma = NaN
        return x == y

    def same_gt(a, b):
        assert set(a) == set(b)
        for k in a:
            if k == "meta":
                assert eq(a[k]["boxes"], b[k]["boxes"]) and eq(a[k].get("steps", []), b[k].get("steps", [])), k
            else:
                assert np.array_equal(a[k], b[k]), k

    # 1 flipping twice is the identity, a sample that is not flipped is untouched, the input arrays are never modified
    g0 = batch["gt"][0]["cur"]
    snap = copy.deepcopy(g0)
    mirrored = flip_batch(copy.deepcopy(batch), [True, False])
    same_gt(batch["gt"][0]["cur"], snap)                                              # (the copy was flipped, not the input)
    again = flip_batch(mirrored, [True, False])
    for key in ("cur", "prev", "next"):
        for i in (0, 1):
            same_gt(again["gt"][i][key], orig["gt"][i][key])
    assert torch.equal(again["omega"], orig["omega"]) and torch.equal(again["radar"]["y"], orig["radar"]["y"])
    assert torch.equal(again["radar"]["psi"], orig["radar"]["psi"])
    assert all(torch.equal(w1, w0) for f1, f0 in zip(again["steps"], orig["steps"]) for (_, _, w1), (_, _, w0) in zip(f1, f0))
    one = flip_batch(copy.deepcopy(batch), [True, False])
    assert torch.equal(one["radar"]["y"][one["radar"]["b"] == 1], orig["radar"]["y"][orig["radar"]["b"] == 1])
    assert torch.equal(one["radar"]["y"][one["radar"]["b"] == 0], -orig["radar"]["y"][orig["radar"]["b"] == 0])
    same_gt(one["gt"][1]["cur"], orig["gt"][1]["cur"])
    # 2 the mirrored dense rasters agree with the mirrored boxes: centroid, axis and velocity of the cells of every box
    checked = 0
    for which, g in (("original", orig["gt"][0]["cur"]), ("mirrored", one["gt"][0]["cur"])):
        res = 0.5
        for i, bx in enumerate(g["meta"]["boxes"]):
            cells = np.argwhere(g["instance"] == i)
            if len(cells) < 25:
                continue
            xs, ys = -0.0 + (cells[:, 0] + 0.5) * res, -26.0 + (cells[:, 1] + 0.5) * res
            if xs.min() < 1 or xs.max() > 59 or abs(ys).max() > 25:
                continue                                                               # cut by the grid border
            assert abs(xs.mean() - bx["center"][0]) < 0.6 and abs(ys.mean() - bx["center"][1]) < 0.6, (which, i)
            ang, ratio = _principal_axis(xs, ys)
            if ratio > 3.0:
                assert abs(math.remainder(ang - bx["yaw"], math.pi)) < 0.25, (which, i, ang, bx["yaw"])
            sp = math.hypot(*bx["vel"])
            if sp > 1.0:
                v = np.median(g["velocity"][:, cells[:, 0], cells[:, 1]].astype(float), axis=1)   # box edge cells are partial
                assert np.allclose(v, bx["vel"], atol=0.05), (which, v, bx["vel"])
            checked += 1
    assert checked >= 6, checked
    # 3 the radar raster of the mirrored detections is the mirrored raster (sin psi and the sub-cell offset change sign)
    B = 2
    R0 = rasterize({k: v for k, v in orig["radar"].items() if k in ("x", "y", "r", "psi", "vr", "rcs", "dt", "b")}, B)
    R1 = rasterize({k: v for k, v in one["radar"].items() if k in ("x", "y", "r", "psi", "vr", "rcs", "dt", "b")}, B)
    want = R0[0].flip(-1).clone()
    want[3] = -want[3]                                                                 # channel 3 = sin psi
    want[-1] = -want[-1]                                                               # last channel = dy
    assert torch.allclose(R1[0], want, atol=1e-5) and torch.allclose(R1[1], R0[1], atol=1e-5)    # not torch.equal: scatter sums on
                                                                                              # several CPU threads are not bit-exact
    # 4 the ego steps: the gap-mean motion mirrors (omega -> -omega), a chain of steps ends at the mirrored pose
    st = orig["gt"][0]["next"]["meta"]["steps"]
    a, b = gap_ego_motion(st), gap_ego_motion(one["gt"][0]["next"]["meta"]["steps"])
    assert abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-6 and abs(a[2] + b[2]) < 1e-9, (a, b)
    assert orig["gap_steps"][0]["next"] == st and one["gap_steps"][0]["next"] == one["gt"][0]["next"]["meta"]["steps"] and \
        one["gap_steps"][1] == orig["gap_steps"][1], "the batch's gap steps mirror with the GT meta steps (and only for the flipped sample)"
    p0 = p1 = torch.zeros(1, 3)
    for (dt, v, w), (_, _, wm) in zip(st, one["gt"][0]["next"]["meta"]["steps"]):
        p0, p1 = advance_pose(p0, dt, v, w), advance_pose(p1, dt, v, wm)
    assert torch.allclose(p1, p0 * torch.tensor([1.0, -1.0, -1.0]), atol=1e-5), (p0, p1)
    # 5 the model: the flip mask mirrors the camera BEV and its visibility, a mask of zeros changes nothing, the loss on the
    #   mirrored GT is finite
    cam = torch.randn(2, 3, 4, 120, 104)
    obs = torch.rand(2, 1, 120, 104)
    c2, o2 = flip_cam(cam, obs, torch.tensor([True, False]))
    assert torch.equal(c2[0], cam[0].flip(-1)) and torch.equal(c2[1], cam[1]) and torch.equal(o2[0], obs[0].flip(-1))
    model = BEVFusionModel(ModelConfig(pretrained="", aux_heads=False)).eval()
    with torch.no_grad():                                      # (the untrained heads output almost constants: compare the trunk)
        y0 = model(copy.deepcopy(orig), dts=(0.0,))["features"]
        b_zero = flip_batch(copy.deepcopy(orig), [False, False])
        y1 = model(b_zero, dts=(0.0,))["features"]
        y2 = model(copy.deepcopy(one), dts=(0.0,))["features"]
    assert torch.allclose(y0, y1, atol=1e-6) and torch.isfinite(y2).all() and (y2[1] - y0[1]).abs().max() < 1e-5
    assert not torch.allclose(y2[0], y0[0], atol=1e-6), float((y2[0] - y0[0]).abs().max())      # the mirrored sample differs
    gts = sample_targets(one["gt"])
    model.train()
    res = model(copy.deepcopy(one), dts=tuple(sorted(gts)))
    L = BEVLoss(grid=model.grid)(res["out"], gts, step_frac=1.0)
    assert torch.isfinite(L["total"])
    print(f"augment selftest ok: double flip = identity, input never modified, {checked} boxes agree with their mirrored rasters "
          f"(centroid, axis, velocity), radar raster of mirrored detections = mirrored raster, ego step / pose mirror, camera BEV "
          f"and visibility flip per sample, zero mask = no change, loss finite on a mirrored batch")


if __name__ == "__main__":
    _selftest()
