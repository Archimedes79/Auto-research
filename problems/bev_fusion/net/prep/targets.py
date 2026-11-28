"""Ground truth -> what the heads and the losses read: the object class groups, the target time of each keyframe of a sample and the head-box targets."""
import math

import numpy as np


NUSC_GROUPS = (("vehicle", "box"), ("two_wheeler", "box"), ("pedestrian", "circle"))   # object head groups (8.3)


SIDE_DT = {"cur": 0.0, "prev": -0.5, "next": 0.5}     # nominal target time of each keyframe of a training sample


def sample_targets(samples):
    """GtGenerator training samples ({'cur', 'prev', 'next'}, a list = a batch) -> {nominal dt: [GT dict per sample]}, keyed by SIDE
    (prev -0.5, cur 0, next +0.5), never by the rounded true gap: every list has one entry per sample, in batch order, aligned
    with the model's predictions at that dt (a 0.75 s keyframe gap, scene-0994, used to round to -1.0 and leave B - 1 targets
    for B predictions). The true gap stays in meta['dt'] / meta['steps']: batches.batch_ego decodes each sample with it."""
    out = {}
    for key, dt in SIDE_DT.items():
        have = [key in s for s in samples]
        if any(have):
            if not all(have):
                raise ValueError(f"sample_targets: '{key}' is missing in {have.count(False)} of {len(samples)} samples (no batch alignment)")
            out[dt] = [s[key] for s in samples]
    return out


def head_box(bx, groups=NUSC_GROUPS):
    """gt_bev meta box (center, size = w, l, h, yaw, vel, group) -> object_head box dict; circles: radius = half the
    larger side. None for a group the head does not have."""
    kinds = dict(groups)
    if bx["group"] not in kinds:
        return None
    w, ln = float(bx["size"][0]), float(bx["size"][1])
    d = {"group": bx["group"], "cx": float(bx["center"][0]), "cy": float(bx["center"][1]),
         "vx": float(bx["vel"][0]), "vy": float(bx["vel"][1])}
    if kinds[bx["group"]] == "box":
        d.update(length=ln, width=w, yaw=float(bx["yaw"]))
    else:
        d["radius"] = max(ln, w) / 2
    return d


def refine_gt_boxes(g, grid, groups=NUSC_GROUPS):
    """GT boxes of one gt_bev GT dict for the S5c candidates (object_head.refine_candidates): head_box dicts whose
    'observed' is True only where stage 1 supervises the box too (a lidar / radar point AND its centre cell covered:
    camera and not occluded, or radar), so every other box is ignored, not background. It does NOT follow
    LossConfig.fusion_area_only."""
    X, Y = np.asarray(g["camera"]).shape[-2:]
    w = np.maximum(np.asarray(g["camera"]) * (1 - np.asarray(g["occluded"])), np.asarray(g["radar"]))
    out = []
    for bx in g.get("meta", {}).get("boxes", []):
        hb = head_box(bx, groups)
        if hb is None:
            continue
        ix = int(math.floor((hb["cx"] - grid.x_min) / grid.res))
        iy = int(math.floor((hb["cy"] - grid.y_min) / grid.res))
        inside = 0 <= ix < X and 0 <= iy < Y
        hb["observed"] = bool(bx.get("observed", True)) and inside and float(w[ix, iy]) > 0
        out.append(hb)
    return out
