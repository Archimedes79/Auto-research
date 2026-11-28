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
Photometric jitter of the camera student's image (t26, TrainConfig.img_jitter / img_jitter_from): draw_jitter makes batch["img_jitter"] (one draw per sample, a
clip is one sample; from
(seed, step, micro-batch) through jitter_seed, so a resume repeats it), the cached model applies jitter_images to the student's input in training mode only.
Tests: tests/test_augment.py
"""

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


# ---- photometric jitter of the camera student's input (t26, TrainConfig.img_jitter): the lift reads cached tables of the camera geometry, so only the colours
# change here
JITTER_RANGE = 0.4          # brightness, contrast and saturation factors: 1 +- 0.4 x strength
JITTER_GAMMA = 1.25         # gamma between 1 / 1.25 and 1.25 at strength 1 (an exponent 1.25 ** (strength u), u in [-1, 1])
JITTER_NOISE = 0.02         # sigma of the gaussian noise in the 0 .. 1 range of the image, x strength
LUMA = (0.299, 0.587, 0.114)


def jitter_seed(seed, step, micro):
    """one integer for (run seed, optimiser step, micro-batch of it): the draw of a step is a function of these three, not of a random stream, so a resume of
    the pass repeats it"""
    return (int(seed) * 1_000_003 + int(step)) * 1_009 + int(micro)


def draw_jitter(B, strength, seed, key=0):
    """the photometric draw of B samples (batch["img_jitter"]): u (4, B) in [-1, 1] = brightness, contrast, saturation, gamma, from `seed` alone (the same for
    every `key`, so the key frames of a clip
    see one scene), and the seed of the pixel noise, which differs per `key` (the key frame of the clip). On the CPU."""
    g = torch.Generator().manual_seed(int(seed))
    return {"u": torch.rand(4, B, generator=g) * 2 - 1, "strength": float(strength), "noise_seed": int(seed) * 64 + int(key)}


def jitter_images(img, j, salt=0):
    """uint8 images (B, 3, H, W) or (B, K, 3, H, W) -> float32 0 .. 255 of the same shape, on the device of `img`: brightness, contrast (about the mean luma of
    the frame), saturation, gamma and gaussian
    noise, in this order, with the factors of sample b from j["u"][:, b] for EVERY frame of the sample (one consistent scene) and the noise drawn per pixel from
    a generator seeded with
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
