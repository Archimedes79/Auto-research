"""Losses of the BEV system (plan.md section 8.4, the loss specification; proposal of 2026-10-26).

    per target time dt: GT of t_c + dt (autogt layout) -> per-layer targets + per-cell weights (masks)
      -> per-output losses: areas, lines (+ centre-lines), dense objects (semantic head), object head stage 1,
         motion head, splat correction penalty, camera / radar auxiliary heads, radar stationary-structure hook
      -> task groups (areas, lines, objects, motion), each term normalised by its own count
    total = sum_g exp(-s_g) L_g + s_g  (Kendall, 1705.07115; or fixed weights)  + fixed-weight auxiliary terms,
    with L_g = sum_dt w(dt) L_g(dt), w(0) = 1, w(+-0.5) = 0.5, w(+-1.0) = 0.25 and a warm-up on the dt != 0 terms.

The six open decisions of 8.4 are options of LossConfig, with the proposal's values as defaults: (1) w_occluded_map,
(2) dense_objects, (3) radar_stationary, (4) use_w_reg (tolerance always; the per-frame weight meta["w_reg"] when the
estimator writes it), (5) dt_weights / warmup_hold / warmup_ramp, (6) task_weighting.
Reused, not duplicated: object_head.build_targets / head_loss (stage 1), timed_head.MotionHead.loss (motion).
The semantic head's channels are declared by layer_spec (SemanticHead(c, len(spec)) gives logits in that order;
distance channels are metres, linear).
Run the self-test: python training/losses.py   (from the bev_fusion folder; CPU, one thread, 1-5 minutes)
"""
import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_BEV_FUSION = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
while not os.path.isdir(os.path.join(_BEV_FUSION, "autogt")) and os.path.dirname(_BEV_FUSION) != _BEV_FUSION:
    _BEV_FUSION = os.path.dirname(_BEV_FUSION)       # a run snapshot (runs/<name>/code/) sits three levels below the bev_fusion folder
if _BEV_FUSION not in sys.path:
    sys.path.append(_BEV_FUSION)                     # the GT layout: the standalone suite bev_fusion/autogt/
from autogt.layers import AREAS, DIST_MAX, LINES, OBJECTS  # noqa: E402

try:
    from .lift_pd import Grid
    from .loss_extras import _wcount, _wsum, boundary_fp_loss, dice_semimetric_loss, ramp
    from .object_head import _gather, build_targets, head_loss, refine_loss
    from .timed_head import MotionHead
except ImportError:                                   # run as a script
    from lift_pd import Grid
    from loss_extras import _wcount, _wsum, boundary_fp_loss, dice_semimetric_loss, ramp
    from object_head import _gather, build_targets, head_loss, refine_loss
    from timed_head import MotionHead

TASK_GROUPS = ("areas", "lines", "objects", "motion")
NUSC_GROUPS = (("vehicle", "box"), ("two_wheeler", "box"), ("pedestrian", "circle"))   # object head groups (8.3)
EPS = 1e-6


@dataclass
class LossConfig:
    # masks (8.4 table)
    w_occluded_map: float = 0.5        # decision (1): map cells behind a GT box (arm 0 / 1)
    w_unknown: float = 0.5             # unknown at the target time (splat mass < mass_threshold, dt != 0)
    mass_threshold: float = 0.5
    w_interpolated: float = 0.5        # interpolated boxes (12 Hz sweeps); the map stays exact
    use_w_reg: bool = True             # decision (4): per-frame registration weight meta["w_reg"] on lines, if present
    # per output
    w_area_iou: float = 0.5
    area_overlap: str = "iou"          # arm "semimetric": Dice semimetric (2303.16296, 0 at p = t for soft targets)
                                       # instead of 1 - soft IoU, on areas and dense objects (loss_extras.py)
    line_tolerance: int = 1            # cells of the dilated Dice
    line_boundary: bool = False        # arm: + ramp(step_frac) x boundary loss on the GT distance field (1812.07032)
    boundary_hold: float = 0.2         # ramp of the boundary term: 0 for this fraction, then linear
    boundary_ramp: float = 0.4
    boundary_w_max: float = 0.5
    huber_delta: float = 0.5           # m
    dist_near: float = 1.0             # m: Huber weight 1 below, dist_w_far above
    dist_w_far: float = 0.25
    dist_max: float = DIST_MAX
    centrelines: bool = False          # mask + direction; off until the S1 baseline exists
    w_direction: float = 0.5
    dense_objects: bool = True         # decision (2): footprint coverage in the semantic head
    w_obj_vel: float = 0.2             # object_head.head_loss velocity weight
    w_obj_mov: float = 1.0             # its moving / static BCE weight (only with model.object_moving, t11)
    w_refine: float = 1.0              # S5c refinement loss in the objects group (when the model refines)
    max_obj: int = 128
    motion_min_speed: float = 1.0
    motion_w_background: float = 0.1
    consistency: bool = False          # object head vel at the GT centres vs the motion field (arm)
    w_consistency: float = 0.1
    w_correction: float = 0.01         # 0.01 |delta|^2 on the splat correction
    w_cam_aux: float = 0.3
    w_radar_aux: float = 0.3
    radar_stationary: bool = True      # decision (3): the proxy target, used when the GT dict carries it
    w_radar_slot: float = 0.1
    # time, decision (5)
    dt_weights: tuple = ((0.0, 1.0), (0.5, 0.5), (1.0, 0.25))       # (|dt| nominal, w), nearest |dt| is used
    warmup_hold: float = 0.2           # dt = 0 only for this fraction of the steps
    warmup_ramp: float = 0.2           # then the dt != 0 terms ramp in linearly over this fraction
    # task groups, decision (6)
    task_weighting: str = "kendall"    # or "fixed"
    fixed_weights: tuple = (1.0, 1.0, 1.0, 1.0)
    s_range: tuple = (-3.0, 1.0)


# ------------------------------------------------------------------------------------------------ layer spec
def layer_spec(centrelines=False, dense_objects=True, stationary=False):
    """Ordered channels of a semantic-type head: list of (channel name, kind, GT key of training/gt_bev.py).
    kinds: area (logit of the coverage), line (logit of the 1-cell mask), dist (truncated distance, metres, linear),
    centreline (logit), dir (cos, sin of the heading; raw, normalised in the loss), object (logit of the footprint
    coverage), stationary (logit of the radar stationary-structure proxy)."""
    spec = [(a, "area", a) for a in AREAS]
    for ln in LINES:
        spec += [(ln, "line", ln)]
    spec += [(ln + "_dist", "dist", ln + "_dist") for ln in LINES]
    if centrelines:
        spec += [("centreline", "centreline", "centreline"), ("centreline_cos", "dir", "centreline_dir"),
                 ("centreline_sin", "dir", "centreline_dir")]
    if dense_objects:
        spec += [(o, "object", o) for o in OBJECTS]
    if stationary:
        spec += [("radar_stationary", "stationary", "radar_stationary")]
    return spec


def radar_aux_spec(stationary=True):
    """radar aux head: dense objects (on radar coverage) + the stationary-structure channel."""
    return [(o, "object", o) for o in OBJECTS] + ([("radar_stationary", "stationary", "radar_stationary")]
                                                  if stationary else [])


def _t(a, device):
    return torch.as_tensor(np.asarray(a, dtype=np.float32), device=device)


def _stack(gts, key, device):
    """(B, 1, X, Y) for (X, Y) layers, (B, K, X, Y) for (K, X, Y) layers."""
    x = torch.stack([_t(g[key], device) for g in gts])
    return x.unsqueeze(1) if x.dim() == 3 else x


def build_layer_targets(gts, cfg=None, mass=None, dt=0.0, device="cpu"):
    """gt_bev GT dicts of one target time (a list, one per batch sample, or one dict) -> dict of tensors:
    every layer by its GT key (B, 1 or K, X, Y), 'obj_cells' (instance >= 0), 'boxes' (meta boxes per sample) and the
    per-cell weights (B, 1, X, Y): w_map (areas: undefined 0, occluded w_occluded_map), w_lines (w_map x w_reg),
    w_obj (objects, velocity: (camera and not occluded) or radar; x w_interpolated on interpolated labels),
    w_cam (camera and not occluded), w_radar (radar); all x w_unknown where mass < threshold at dt != 0."""
    cfg = cfg or LossConfig()
    gts = [gts] if isinstance(gts, dict) else list(gts)
    T = {}
    keys = list(AREAS) + list(LINES) + [ln + "_dist" for ln in LINES] + list(OBJECTS) + ["velocity"]
    if cfg.centrelines:
        keys += ["centreline", "centreline_dir"]
    for k in keys:
        T[k] = _stack(gts, k, device)
    if cfg.radar_stationary and all("radar_stationary" in g for g in gts):       # hook: another session builds it
        T["radar_stationary"] = _stack(gts, "radar_stationary", device)
    T["obj_cells"] = (_stack(gts, "instance", device) >= 0).float()
    cam, occ = _stack(gts, "camera", device), _stack(gts, "occluded", device)
    radar, undef = _stack(gts, "radar", device), _stack(gts, "undefined", device)
    w_map = (1 - undef) * torch.where(occ > 0, torch.full_like(occ, cfg.w_occluded_map), torch.ones_like(occ))
    w_cam = cam * (1 - occ)
    w_obj = torch.maximum(w_cam, radar)
    interp = []
    for g in gts:                                     # cell flag or frame flag; gt_bev keyframes carry neither (exact)
        f = g.get("interpolated", g.get("meta", {}).get("interpolated", 0.0))
        interp.append(_t(f, device) * torch.ones_like(w_obj[0, 0]))
    interp = torch.stack(interp).unsqueeze(1)
    w_obj = w_obj * torch.where(interp > 0, torch.full_like(interp, cfg.w_interpolated), torch.ones_like(interp))
    # boxes without any lidar / radar point (gt_bev 'ignore' layer, meta box observed = False): footprint ignored in
    # every object-type target (dense layers, velocity / motion, aux heads), never background (plan 8.4 masks)
    keep = (1 - _stack(gts, "ignore", device).clamp(0, 1)) if all("ignore" in g for g in gts) else torch.ones_like(cam)
    w_obj = w_obj * keep
    w_reg = torch.tensor([float(g.get("meta", {}).get("w_reg", 1.0)) if cfg.use_w_reg else 1.0 for g in gts],
                         device=device).view(-1, 1, 1, 1)
    W = {"w_map": w_map, "w_lines": w_map * w_reg, "w_obj": w_obj, "w_cam": w_cam, "w_radar": radar,
         "w_keep": keep}
    if mass is not None and abs(float(dt)) > 1e-6:
        m = mass.detach().to(device)
        u = torch.where(m < cfg.mass_threshold, torch.full_like(m, cfg.w_unknown), torch.ones_like(m))
        W = {k: v * u for k, v in W.items()}
    T.update(W)
    T["boxes"] = [g.get("meta", {}).get("boxes", []) for g in gts]
    return T


# ------------------------------------------------------------------------------------------------ per-output losses
def soft_bce(logits, target, w):
    """BCE on soft targets in [0, 1], weighted mean per channel -> (K,)."""
    return _wsum(F.binary_cross_entropy_with_logits(logits, target, reduction="none"), w) / _wcount(w, logits)


def soft_iou_loss(prob, target, w):
    """1 - soft IoU per channel -> (K,); 0 for a channel without GT (false positives there are the BCE's job)."""
    i, u = _wsum(prob * target, w), _wsum(prob + target - prob * target, w)
    return torch.where(_wsum(target, w) > 0, 1 - i / u.clamp(min=EPS), torch.zeros_like(i))


def tolerant_dice_loss(prob, target, w, tol=1):
    """Dice with a tolerance of tol cells -> (K,): 1 - (sum P dil(G) + sum G dil(P)) / (sum P + sum G) (weighted), so a
    line found within tol cells costs nothing, one further away does; tol = 0 is the plain soft Dice. Cells of weight
    0 are zeroed before the dilation (no gradient leaks into them); 0 for a channel without GT."""
    m = (w > 0).to(prob.dtype)
    p, g = prob * m, target * m
    if tol > 0:
        k = 2 * tol + 1
        dp, dg = F.max_pool2d(p, k, 1, tol), F.max_pool2d(g, k, 1, tol)
    else:
        dp, dg = p, g
    num, den = _wsum(p * dg, w) + _wsum(g * dp, w), _wsum(p, w) + _wsum(g, w)
    return torch.where(_wsum(g, w) > 0, 1 - num / den.clamp(min=EPS), torch.zeros_like(num))


def dist_huber(pred, target, w, delta=0.5, near=1.0, w_far=0.25, dist_max=DIST_MAX):
    """Huber (delta, metres) on the truncated distance field -> (K,); cell weight 1 below `near`, w_far above; where the
    GT is truncated (>= dist_max) a prediction beyond dist_max is not penalised."""
    p = torch.where(target >= dist_max - 1e-4, pred.clamp(max=dist_max), pred)
    ww = w * torch.where(target < near, torch.ones_like(target), torch.full_like(target, w_far))
    return _wsum(F.huber_loss(p, target, delta=delta, reduction="none"), ww) / ww.sum((0, 2, 3)).clamp(min=EPS)


def direction_l1(pred, target, mask, w):
    """L1 on the unit direction (cos, sin) at the mask cells; pred (B, 2, X, Y) raw, normalised here."""
    u = pred / pred.norm(dim=1, keepdim=True).clamp(min=EPS)
    ww = w * mask
    return ((u - target).abs() * ww).sum() / (2 * ww.sum()).clamp(min=EPS)


def correction_penalty(delta):
    """mean |delta|^2 of the splat correction (..., 2), metres."""
    return (delta ** 2).sum(-1).mean()


def overlap_loss(prob, target, w, cfg):
    """The region-overlap term of areas and dense objects: 1 - soft IoU, or the Dice semimetric arm."""
    if cfg.area_overlap == "semimetric":
        return dice_semimetric_loss(prob, target, w)
    if cfg.area_overlap != "iou":
        raise ValueError(cfg.area_overlap)
    return soft_iou_loss(prob, target, w)


def map_terms(logits, T, spec, weights, cfg, step_frac=1.0):
    """Losses of a semantic-type head (main or auxiliary) whose channels follow `spec`. weights: kind group -> (B, 1,
    X, Y) ('area', 'line', 'object', 'stationary'). Returns (terms, groups): per-channel terms and the group sums
    (per-channel terms averaged inside a group, so a group is of order 1). step_frac: training progress (ramps)."""
    idx = {}
    for i, (_, kind, _) in enumerate(spec):
        idx.setdefault(kind, []).append(i)
    names = {k: [spec[i][0] for i in v] for k, v in idx.items()}

    def ch(kind):
        return logits[:, idx[kind]]

    def tg(kind):
        return torch.cat([T[spec[i][2]] for i in idx[kind]], 1)
    terms, groups = {}, {}
    if "area" in idx:
        x, t, w = ch("area"), tg("area"), weights["area"]
        bce, iou = soft_bce(x, t, w), overlap_loss(torch.sigmoid(x), t, w, cfg)
        for j, n in enumerate(names["area"]):
            terms[f"area_bce.{n}"], terms[f"area_iou.{n}"] = bce[j], iou[j]
        groups["areas"] = (bce + cfg.w_area_iou * iou).mean()
    if "line" in idx:
        w = weights["line"]
        dice = tolerant_dice_loss(torch.sigmoid(ch("line")), tg("line"), w, cfg.line_tolerance)
        hub = dist_huber(ch("dist"), tg("dist"), w, cfg.huber_delta, cfg.dist_near, cfg.dist_w_far, cfg.dist_max)
        for j, n in enumerate(names["line"]):
            terms[f"line_dice.{n}"] = dice[j]
        for j, n in enumerate(names["dist"]):
            terms[f"line_huber.{n}"] = hub[j]
        groups["lines"] = dice.mean() + hub.mean()
        if cfg.line_boundary:
            wb = ramp(step_frac, cfg.boundary_hold, cfg.boundary_ramp, cfg.boundary_w_max)
            bnd = boundary_fp_loss(torch.sigmoid(ch("line")), tg("dist"), w, cfg.dist_max)
            for j, n in enumerate(names["line"]):
                terms[f"line_boundary.{n}"] = bnd[j]
            groups["lines"] = groups["lines"] + wb * bnd.mean()
    if "centreline" in idx:
        w = weights["line"]
        dice = tolerant_dice_loss(torch.sigmoid(ch("centreline")), tg("centreline"), w, cfg.line_tolerance)[0]
        dl = direction_l1(ch("dir"), T["centreline_dir"], T["centreline"], w)
        terms["centreline_dice"], terms["centreline_dir"] = dice, dl
        groups["lines"] = groups.get("lines", 0.0) + dice + cfg.w_direction * dl
    if "object" in idx:
        x, t, w = ch("object"), tg("object"), weights["object"]
        bce, iou = soft_bce(x, t, w), overlap_loss(torch.sigmoid(x), t, w, cfg)
        for j, n in enumerate(names["object"]):
            terms[f"obj_bce.{n}"], terms[f"obj_iou.{n}"] = bce[j], iou[j]
        groups["objects"] = (bce + iou).mean()
    if "stationary" in idx and "radar_stationary" in T:          # hook: skipped while the GT has no such layer
        x, t, w = ch("stationary"), T["radar_stationary"], weights["stationary"]
        bce = soft_bce(x, t, w)[0]
        dice = tolerant_dice_loss(torch.sigmoid(x), t, w, 0)[0]
        terms["stationary_bce"], terms["stationary_dice"] = bce, dice
        groups["stationary"] = bce + dice
    return terms, groups


def head_box(bx, groups=NUSC_GROUPS):
    """gt_bev meta box (center, size = w, l, h, yaw, vel, group) -> object_head box dict; circles: radius = half the
    larger side (as tests/test_object_head.py). None for a group the head does not have."""
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
    camera and not occluded, or radar), so every other box is ignored, not background."""
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


def object_head_terms(out, boxes_batch, w_obj, grid, groups=NUSC_GROUPS, w_vel=0.2, max_obj=128, w_mov=1.0):
    """Stage 1 via object_head.head_loss. A box is supervised if it is observed (gt_bev meta 'observed': any lidar /
    radar point) and its centre cell has w_obj > 0; the others are ignored (neither positive nor negative: their peak
    region and every cell of w_obj = 0 get a constant logit, so no loss gradient). Returns (head_loss dict, targets)."""
    X, Y = w_obj.shape[-2:]
    keep, drop = [], []
    for b, boxes in enumerate(boxes_batch):
        k, d = [], []
        for bx in boxes:
            hb = head_box(bx, groups)
            if hb is None:
                continue
            ix = int(math.floor((hb["cx"] - grid.x_min) / grid.res))       # = build_targets' centre cell
            iy = int(math.floor((hb["cy"] - grid.y_min) / grid.res))
            if not (0 <= ix < X and 0 <= iy < Y):
                continue
            (k if bx.get("observed", True) and float(w_obj[b, 0, ix, iy]) > 0 else d).append(hb)
        keep.append(k)
        drop.append(d)
    dev = out["hm"].device
    t = build_targets(keep, grid, groups, max_obj)
    ign = (w_obj <= 0).expand_as(out["hm"])
    if any(drop):
        ign = ign | ((build_targets(drop, grid, groups, max_obj)["hm"].to(dev) > 0) & (t["hm"].to(dev) == 0))
    hm = torch.where(ign, torch.full_like(out["hm"], -1e4), out["hm"])
    return head_loss(dict(out, hm=hm), t, w_vel, w_mov=w_mov), t


def motion_term(v_hat, T, cfg):
    """MotionHead.loss on the cells where objects / velocity are supervised (w_obj > 0), normalised by their counts."""
    valid = T["w_obj"][:, 0] > 0

    def sel(x):
        return x.permute(0, 2, 3, 1)[valid].t().reshape(1, x.shape[1], -1, 1)
    return MotionHead.loss(sel(v_hat), sel(T["velocity"]), sel(T["obj_cells"]), cfg.motion_min_speed,
                           cfg.motion_w_background)


def consistency_term(obj_vel, v_hat, t):
    """L1 between the object head's vel and the motion field at the GT centres (8.2 / 8.4 optional)."""
    m = t["mask"].to(v_hat.device).unsqueeze(-1)
    ind = t["ind"].to(v_hat.device)
    return ((_gather(obj_vel, ind) - _gather(v_hat, ind)).abs() * m).sum() / (2 * m.sum()).clamp(min=1)


# ------------------------------------------------------------------------------------------------ time, task groups
def dt_weight(dt, step_frac=1.0, cfg=None):
    """w(dt) of the nearest nominal |dt|, times the warm-up ramp for dt != 0 (0 for step_frac < hold, then linear). A |dt| with no
    listed weight within 0.1 s is refused: `[[0, 1]]` alone used to give dt = +-0.5 the weight 1 of dt = 0 (review 2026-10-31 pass 2)."""
    cfg = cfg or LossConfig()
    near, w = min(cfg.dt_weights, key=lambda kv: abs(abs(float(dt)) - kv[0]))
    if abs(abs(float(dt)) - near) > 0.1:
        raise ValueError(f"dt_weights {cfg.dt_weights} list no weight for |dt| = {abs(float(dt)):.2f} (nearest: {near})")
    if abs(float(dt)) > 1e-6:
        w *= min(1.0, max(0.0, (step_frac - cfg.warmup_hold) / max(cfg.warmup_ramp, EPS)))
    return w


SIDE_DT = {"cur": 0.0, "prev": -0.5, "next": 0.5}     # nominal target time of each keyframe of a training sample


def sample_targets(samples):
    """autogt training samples ({'cur', 'prev', 'next'}, a list = a batch) -> {nominal dt: [GT dict per sample]}, keyed by SIDE
    (prev -0.5, cur 0, next +0.5), never by the rounded true gap: every list has one entry per sample, in batch order, aligned
    with the model's predictions at that dt (a 0.75 s keyframe gap, scene-0994, used to round to -1.0 and leave B - 1 targets
    for B predictions). The true gap stays in meta['dt'] / meta['steps']: train.batch_ego decodes each sample with it."""
    out = {}
    for key, dt in SIDE_DT.items():
        have = [key in s for s in samples]
        if any(have):
            if not all(have):
                raise ValueError(f"sample_targets: '{key}' is missing in {have.count(False)} of {len(samples)} samples (no batch alignment)")
            out[dt] = [s[key] for s in samples]
    return out


class TaskWeights(nn.Module):
    """Uncertainty weights between the task groups (Kendall et al. 1705.07115): sum_g exp(-s_g) L_g + s_g, s_g =
    log sigma_g^2 start 0, kept inside s_range (projected before every use, the gradient passes inside the range);
    task_weighting 'fixed': sum_g c_g L_g."""

    def __init__(self, groups=TASK_GROUPS, mode="kendall", fixed=(1.0, 1.0, 1.0, 1.0), s_range=(-3.0, 1.0)):
        super().__init__()
        self.groups, self.mode, self.fixed, self.s_range = tuple(groups), mode, tuple(fixed), tuple(s_range)
        self.s = nn.Parameter(torch.zeros(len(self.groups)))

    def values(self):
        with torch.no_grad():
            self.s.clamp_(*self.s_range)
        return self.s.clamp(*self.s_range)

    def forward(self, group_losses):
        """group_losses: name -> scalar (missing groups are left out) -> (total, {name: s or weight})."""
        if self.mode == "fixed":
            used = {g: c for g, c in zip(self.groups, self.fixed) if g in group_losses}
            return sum(c * group_losses[g] for g, c in used.items()), used
        s = self.values()
        total, used = 0.0, {}
        for i, g in enumerate(self.groups):
            if g in group_losses:
                total = total + torch.exp(-s[i]) * group_losses[g] + s[i]
                used[g] = s[i].detach()
        return total, used


class BEVLoss(nn.Module):
    """The 8.4 total. forward(preds, gts, step_frac):
    preds: {dt: {'sem': (B, len(spec), X, Y) logits, 'obj': object head maps, 'v_hat': (B, 2, X, Y) motion field
            (supervised at dt = 0 against the GT of t_c), 'mass': (B, 1, X, Y) splat mass at dt, 'delta': (..., 2)
            splat correction, 'cam_aux': (B, len(cam_spec), X, Y), 'radar_aux': (B, len(radar_spec), X, Y),
            'radar_slot': a precomputed radar slot-decoder loss}; every entry optional}
    gts: {dt: [gt_bev GT dict per sample]} (sample_targets builds it from training samples)
    Returns a dict: every term as '<name>@<dt>', 'group.<g>' (dt-weighted), 's.<g>', 'aux', 'total'."""

    def __init__(self, cfg=None, grid=None, groups=NUSC_GROUPS):
        super().__init__()
        self.cfg, self.grid, self.groups = cfg or LossConfig(), grid or Grid(), groups
        c = self.cfg
        self.spec = layer_spec(c.centrelines, c.dense_objects)
        self.cam_spec = layer_spec(False, True)
        self.radar_spec = radar_aux_spec(c.radar_stationary)
        self.task = TaskWeights(TASK_GROUPS, c.task_weighting, c.fixed_weights, c.s_range)

    def terms_at(self, p, gts, dt, step_frac=1.0):
        """(terms, task-group losses, fixed-weight auxiliary losses) of one target time."""
        cfg = self.cfg
        dev = next(v for v in p.values() if torch.is_tensor(v)) if "obj" not in p else p["obj"]["hm"]
        T = build_layer_targets(gts, cfg, p.get("mass"), dt, dev.device)
        terms, groups, fixed = {}, {}, {}

        def add(g, v):
            groups[g] = groups.get(g, 0.0) + v
        if "sem" in p:
            w = {"area": T["w_map"], "line": T["w_lines"], "object": T["w_obj"]}
            t, g = map_terms(p["sem"], T, self.spec, w, cfg, step_frac)
            terms.update(("sem." + k, v) for k, v in t.items())
            for k, v in g.items():
                add(k, v)
        tt = None
        if "obj" in p:
            lo, tt = object_head_terms(p["obj"], T["boxes"], T["w_obj"], self.grid, self.groups, cfg.w_obj_vel,
                                       cfg.max_obj, cfg.w_obj_mov)
            terms.update(("objhead." + k, v) for k, v in lo.items())
            add("objects", lo["total"])
        if "refine" in p:                                     # S5c, candidates built by the model (model.refine)
            r = p["refine"]                                   # tensors only (the trainer casts every entry to float)
            steps = [{k: r[k][i] for k in ("box_in", "delta", "quality")} for i in range(r["delta"].shape[0])]
            lr = refine_loss(steps, r["gt"], r["matched"], r["valid"], r["group"].long(), self.groups, cfg.w_obj_vel)
            terms["refine.reg"], terms["refine.quality"] = lr["reg"], lr["quality"]
            add("objects", cfg.w_refine * lr["total"])
        if "v_hat" in p and abs(float(dt)) < 1e-6:
            terms["motion"] = motion_term(p["v_hat"], T, cfg)
            lv = [motion_term(v, T, cfg) for v in (p.get("v_levels") or {}).values()]
            for i, t in enumerate(lv):                           # t15: the motion block's coarser levels (deep supervision, as in t12)
                terms[f"motion_l{i}"] = t
            add("motion", (terms["motion"] + sum(lv)) / (1 + len(lv)))   # the mean over the levels: the group keeps the weight of one head
            if cfg.consistency and tt is not None:
                terms["motion_consistency"] = consistency_term(p["obj"]["vel"], p["v_hat"], tt)
                add("motion", cfg.w_consistency * terms["motion_consistency"])
        if "delta" in p:
            terms["correction"] = correction_penalty(p["delta"])
            fixed["correction"] = cfg.w_correction * terms["correction"]
        if "cam_aux" in p:
            wc = T["w_cam"]
            t, g = map_terms(p["cam_aux"], T, self.cam_spec, {"area": wc, "line": wc, "object": wc * T["w_keep"]},
                             cfg, step_frac)
            terms.update(("cam_aux." + k, v) for k, v in t.items())
            fixed["cam_aux"] = cfg.w_cam_aux * sum(g.values())
        if "radar_aux" in p:
            wr = T["w_radar"]
            t, g = map_terms(p["radar_aux"], T, self.radar_spec, {"object": wr * T["w_keep"], "stationary": wr}, cfg,
                             step_frac)
            terms.update(("radar_aux." + k, v) for k, v in t.items())
            fixed["radar_aux"] = cfg.w_radar_aux * sum(g.values())
        if "radar_slot" in p:
            terms["radar_slot"] = p["radar_slot"]
            fixed["radar_slot"] = cfg.w_radar_slot * p["radar_slot"]
        return terms, groups, fixed

    def forward(self, preds, gts, step_frac=1.0):
        out, group_tot, aux = {}, {}, 0.0
        for dt, p in preds.items():
            w = dt_weight(dt, step_frac, self.cfg)
            if w <= 0:
                continue
            terms, groups, fixed = self.terms_at(p, gts[dt], dt, step_frac)
            tag = f"@{float(dt):+.1f}"
            out.update((k + tag, v) for k, v in terms.items())
            for g, v in groups.items():
                group_tot[g] = group_tot.get(g, 0.0) + w * v
            for k, v in fixed.items():
                aux = aux + w * v
        total, s = self.task(group_tot)
        out.update(("group." + g, v) for g, v in group_tot.items())
        out.update(("s." + g, v) for g, v in s.items())
        out["aux"] = aux
        out["total"] = total + aux
        return out


# ----------------------------------------------------------------------------------------------- self-test
def synthetic_gt(scene, grid):
    """A GT dict in the gt_bev layout from a synthetic-world scene (tests/synthetic_world.py): road band = drivable
    area, inner lines = lane dividers, outer lines = road dividers, lane centres = centre-lines, a walkway band, boxes;
    camera fan +-50 deg from (1.5, 0), radar fan +-60 deg from (3.5, 0) up to 70 m; radar_stationary = parked boxes +
    a 1 m band inside the drivable edge (the 8.4 proxy)."""
    import synthetic_world as sw
    from autogt.layers import cell_centres, occlusion
    gx, gy = cell_centres(grid)
    base = sw.gt_bev(scene, grid)
    x0, y0, yaw = scene.lane_pose
    yl = -math.sin(yaw) * (gx - x0) + math.cos(yaw) * (gy - y0)
    edge = max(abs(v) for v in scene.lanes) + 1.5
    f32 = np.float32
    g = {"drivable_area": (np.abs(yl) <= edge).astype(f32),
         "walkway": ((np.abs(yl) > edge) & (np.abs(yl) <= edge + 3)).astype(f32)}
    for k in ("ped_crossing", "stop_line", "carpark_area", "two_wheeler"):
        g[k] = np.zeros(gx.shape, f32)
    inner = [v for v in scene.lanes if abs(v) < 3]
    outer = [v for v in scene.lanes if abs(v) >= 3]
    for name, ys in (("lane_divider", inner), ("road_divider", outer)):
        d = np.minimum(np.min([np.abs(yl - v) for v in ys], 0), DIST_MAX).astype(f32)
        g[name + "_dist"], g[name] = d, (d <= grid.res / 2).astype(f32)
    ls = sorted(scene.lanes)
    cl = np.min([np.abs(yl - (a + b) / 2) for a, b in zip(ls[:-1], ls[1:])], 0) <= grid.res / 2
    g["centreline"] = cl.astype(f32)
    g["centreline_dir"] = np.stack([np.full(gx.shape, math.cos(yaw)), np.full(gx.shape, math.sin(yaw))]).astype(f32) * cl
    g["vehicle"], g["pedestrian"] = base["vehicles"].numpy(), base["pedestrians"].numpy()
    g["velocity"], g["instance"] = base["velocity"].numpy(), base["box_id"].numpy()
    cam = (gx > 2.0) & (np.abs(np.arctan2(gy, gx - 1.5)) < math.radians(50))
    rad = (np.hypot(gx - 3.5, gy) <= 70) & (np.abs(np.arctan2(gy, gx - 3.5)) <= math.radians(60))
    fps = [np.array(b.corners()) for b in scene.boxes]
    g["camera"], g["radar"] = cam.astype(f32), rad.astype(f32)
    g["occluded"] = occlusion(gx, gy, (1.5, 0.0), fps).astype(f32)
    g["undefined"] = (~cam & ~rad).astype(f32)
    parked = np.zeros(gx.shape, bool)
    for k, b in enumerate(scene.boxes):
        if math.hypot(b.vx, b.vy) < 0.5:
            parked |= g["instance"] == k
    g["radar_stationary"] = (parked | ((np.abs(yl) <= edge) & (np.abs(yl) > edge - 1))).astype(f32)
    g["meta"] = {"res": grid.res, "boxes": [
        {"group": "pedestrian" if b.cls == "pedestrian" else "vehicle", "corners": fp,
         "center": np.array([b.cx, b.cy, 0.0]), "size": [b.width, b.length, b.height], "yaw": b.heading,
         "vel": np.array([b.vx, b.vy])} for b, fp in zip(scene.boxes, fps)]}
    return g


def _selftest():
    import os
    import sys
    import time
    from dataclasses import replace
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
    import synthetic_world as sw
    from bev_trunk import SemanticHead
    from object_head import ObjectHead
    from timed_head import TimedHead, TimeMover
    torch.manual_seed(0)
    torch.set_num_threads(1)                          # small tensors on a shared CPU: threads only add contention
    t0 = time.time()
    grid, cfg = Grid(), LossConfig()
    X, Y = grid.nx, grid.ny
    spec = layer_spec(cfg.centrelines, cfg.dense_objects)
    print("layer spec (semantic head channels):", ", ".join(f"{i}:{n}" for i, (n, _, _) in enumerate(spec)))
    ones = torch.ones(1, 1, X, Y)

    # 1 BCE on soft targets = the analytic value (also with arbitrary weights)
    z, tv = math.log(3.0), 0.25
    exp = -(tv * math.log(0.75) + (1 - tv) * math.log(0.25))
    w = torch.rand(1, 1, X, Y)
    got = soft_bce(torch.full((1, 2, X, Y), z), torch.full((1, 2, X, Y), tv), w)
    assert torch.allclose(got, torch.tensor(exp), atol=1e-5), (got, exp)
    # 2 tolerant Dice: 0 for a 1-cell shift (either side), > 0 for a 3-cell shift
    G = torch.zeros(1, 1, X, Y)
    G[0, 0, 10:110, 50] = 1.0
    for s, zero in ((1, True), (-1, True), (0, True), (3, False)):
        P = torch.roll(G, s, 3)
        d = float(tolerant_dice_loss(P, G, ones, 1)[0])
        assert (abs(d) < 1e-6) if zero else d > 0.9, (s, d)
    d3 = float(tolerant_dice_loss(torch.roll(G, 3, 3), G, ones, 1)[0])
    # 3 Huber 0 for the exact field; > 0 for a shifted one; a prediction beyond the truncation is free where GT = d_max
    gt0 = synthetic_gt(sw.Scene([]), grid)
    D = torch.as_tensor(gt0["lane_divider_dist"])[None, None]
    assert float(dist_huber(D, D, ones)[0]) == 0.0
    assert float(dist_huber(D + 0.3, D, ones)[0]) > 0.0
    assert float(dist_huber(torch.where(D >= DIST_MAX, D + 5, D), D, ones)[0]) == 0.0

    # the synthetic sample: one moving car, a parked car, a parked truck, a walking pedestrian; GT at dt = 0, +-0.5 s
    B_ = sw.Box
    scene = sw.Scene([B_(20.0, -1.75, 4.5, 1.8, 1.5, 0.0, 8.0, 0.0, "car", 10.0),
                      B_(28.0, -5.5, 4.5, 1.8, 1.5, 0.0, 0.0, 0.0, "car", 10.0),
                      B_(38.0, 3.5, 8.0, 2.5, 3.0, 0.05, 0.0, 0.0, "truck", 20.0),
                      B_(15.0, 8.5, 0.6, 0.6, 1.8, math.pi / 2, 0.0, 1.2, "pedestrian", -5.0)],
                     lanes=[-5.25, -1.75, 1.75, 5.25], v=10.0, omega=0.1)
    dts = (0.0, 0.5, -0.5)
    gts = {dt: [synthetic_gt(scene.step(dt) if dt else scene, grid)] for dt in dts}
    T = build_layer_targets(gts[0.0], cfg)
    assert T["drivable_area"].shape == (1, 1, X, Y) and T["velocity"].shape == (1, 2, X, Y)
    assert float(T["w_map"].min()) == 0.0 and float(T["w_map"].max()) == 1.0
    assert abs(float(T["w_map"][T["w_map"] > 0].min()) - cfg.w_occluded_map) < 1e-6, "occluded map cells: 0.5"
    assert "radar_stationary" in T
    gno = {k: v for k, v in gts[0.0][0].items() if k != "radar_stationary"}
    assert "radar_stationary" not in build_layer_targets([gno], cfg), "hook skipped when the GT has no such layer"
    mass = torch.ones(1, 1, X, Y)
    mass[..., :20, :] = 0.0
    Tu = build_layer_targets(gts[0.5], cfg, mass, 0.5)
    Tk = build_layer_targets(gts[0.5], cfg, None, 0.5)
    sel = Tk["w_map"][..., :20, :] > 0
    assert torch.allclose(Tu["w_map"][..., :20, :][sel], cfg.w_unknown * Tk["w_map"][..., :20, :][sel])

    # predictions as leaf tensors: weight-0 cells get exactly zero gradient; doubling the map leaves every term
    cam_n, rad_n = len(layer_spec(False, True)), len(radar_aux_spec(True))

    def leaf_preds(Xn, seed):
        g = torch.Generator().manual_seed(seed)
        r = lambda *s: torch.randn(*s, generator=g)
        p = {}
        for dt in (0.0, 0.5):
            q = {"sem": r(1, len(spec), Xn, Y), "cam_aux": r(1, cam_n, Xn, Y), "radar_aux": r(1, rad_n, Xn, Y),
                 "obj": {k: r(1, c, Xn, Y) for k, c in (("hm", 3), ("off", 2), ("size", 2), ("rot", 2), ("rad", 1),
                                                        ("vel", 2))},
                 "delta": 0.1 * r(1, 20, 2)}
            if dt == 0.0:
                q["v_hat"] = r(1, 2, Xn, Y)
            else:
                q["mass"] = torch.rand(1, 1, Xn, Y, generator=g)
            p[dt] = q
        return p

    def leaves(p):
        for q in p.values():
            for k, v in q.items():
                for t_ in (v.values() if isinstance(v, dict) else [v]):
                    if k != "mass":
                        yield k, t_
    gz = {dt: [dict(gts[dt][0])] for dt in (0.0, 0.5)}
    for dt in gz:                                     # an undefined block (y < -21 m) and undefined first / last rows
        g = gz[dt][0] = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in gz[dt][0].items()}
        for k in ("camera", "radar", "occluded"):
            g[k][:, :10] = 0
            g[k][[0, -1], :] = 0
        g["undefined"][:, :10] = 1
        g["undefined"][[0, -1], :] = 1
    lossf = BEVLoss(cfg, grid)
    p = leaf_preds(X, 1)
    for _, t_ in leaves(p):
        t_.requires_grad_(True)
    res = lossf(p, gz, 1.0)
    res["total"].backward()
    for k, t_ in leaves(p):
        if t_.dim() == 4:
            assert float(t_.grad[..., :10].abs().max()) == 0.0, f"gradient in weight-0 cells: {k}"
            assert float(t_.grad[..., [0, -1], :].abs().max()) == 0.0, f"gradient in weight-0 rows: {k}"
            if k in ("sem", "cam_aux", "radar_aux", "hm", "v_hat"):
                assert float(t_.grad.abs().max()) > 0, f"no gradient at all: {k}"
    assert lossf.task.s.grad is not None and float(lossf.task.s.grad.abs().min()) > 0, "gradient reaches every s_g"

    span = grid.x_max - grid.x_min
    grid2 = replace(grid, x_max=grid.x_max + span)

    def doubled(g):
        o = {}
        for k, v in g.items():
            if k == "meta":
                bx2 = [dict(b, center=b["center"] + np.array([span, 0, 0]), corners=b["corners"] + [span, 0])
                       for b in v["boxes"]]
                o[k] = dict(v, boxes=v["boxes"] + bx2)
            else:
                o[k] = np.concatenate([v, v], axis=v.ndim - 2)
        return o
    p1 = leaf_preds(X, 2)
    p2 = {dt: {k: ({kk: torch.cat([vv, vv], 2) for kk, vv in v.items()} if isinstance(v, dict) else
                   torch.cat([v, v], 2) if v.dim() == 4 else torch.cat([v, v], 1)) for k, v in q.items()}
          for dt, q in p1.items()}
    with torch.no_grad():
        r1 = BEVLoss(cfg, grid)(p1, gz, 1.0)
        r2 = BEVLoss(cfg, grid2)(p2, {dt: [doubled(gz[dt][0])] for dt in gz}, 1.0)
    assert set(r1) == set(r2)
    worst = max(abs(float(r1[k]) - float(r2[k])) / max(1.0, abs(float(r1[k]))) for k in r1)
    assert worst < 1e-4, [(k, float(r1[k]), float(r2[k])) for k in r1 if abs(float(r1[k]) - float(r2[k])) > 1e-4]

    # Kendall weights stay inside the clamp under strong pressure either way; fixed weights; dt weights and warm-up
    tw = TaskWeights()
    opt = torch.optim.SGD(tw.parameters(), lr=5.0)
    for i in range(40):
        big = 1e3 if i < 20 else 1e-6
        tot, s = tw({g: torch.tensor(big) for g in TASK_GROUPS})
        opt.zero_grad()
        tot.backward()
        opt.step()
        v = tw.values().detach()
        assert float(v.min()) >= -3.0 and float(v.max()) <= 1.0
    assert abs(float(tw.values().detach()[0]) + 3.0) < 1e-6, "pushed to the lower bound"
    tf, _ = TaskWeights(mode="fixed", fixed=(1, 2, 3, 4))({"areas": torch.tensor(1.0), "motion": torch.tensor(1.0)})
    assert float(tf) == 5.0
    assert dt_weight(0.0, 0.0) == 1.0 and dt_weight(0.5, 0.1) == 0.0 and abs(dt_weight(-0.5, 0.3) - 0.25) < 1e-9
    assert dt_weight(0.46, 0.5) == 0.5 and dt_weight(-1.0, 1.0) == 0.25

    # overfit one synthetic sample: a free state, ONE shared mover, the semantic, object and both auxiliary heads as
    # TimedHeads at dt = 0, +-0.5 s, a free correction delta; ~100 Adam steps; the total is finite and decreases
    torch.manual_seed(0)
    C = 16
    state = nn.Parameter(0.1 * torch.randn(1, C, X, Y))
    delta = nn.Parameter(0.3 * torch.ones(1, 8, 2))
    mover = TimeMover(C, grid)
    sem = TimedHead(SemanticHead(C, len(spec)), C, mover)
    obj = TimedHead(ObjectHead(C, NUSC_GROUPS, hidden=16), C, mover)                 # small: CPU test
    cam = TimedHead(SemanticHead(C, cam_n), C, mover)
    rad = TimedHead(SemanticHead(C, rad_n), C, mover)
    lossf = BEVLoss(cfg, grid)
    params = [state, delta] + list(lossf.parameters())
    for m in (sem, obj, cam, rad):
        params += [q for q in m.parameters() if all(q is not r_ for r_ in params)]
    opt = torch.optim.Adam(params, lr=1e-2)
    hist = []
    for it in range(100):
        preds = {}
        for dt in dts:
            y, aux = sem(state, dt, scene.v, scene.omega, return_aux=True)
            q = {"sem": y, "mass": aux["mass"], "obj": obj(state, dt, scene.v, scene.omega),
                 "cam_aux": cam(state, dt, scene.v, scene.omega), "radar_aux": rad(state, dt, scene.v, scene.omega)}
            if dt == 0.0:
                q["v_hat"] = aux["v_hat"]
            else:
                q["delta"] = delta
            preds[dt] = q
        res = lossf(preds, gts, 1.0)
        assert torch.isfinite(res["total"]), it
        raw = sum(res["group." + g].item() for g in TASK_GROUPS) + res["aux"].item()
        hist.append((res["total"].item(), raw))
        opt.zero_grad()
        res["total"].backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
    (t_first, r_first), (t_last, r_last) = hist[0], hist[-1]
    assert t_last < t_first and r_last < 0.7 * r_first, hist[::10]
    assert correction_penalty(delta).item() < 0.3 ** 2 * 2
    # an unobserved box (no lidar / radar point): not a heatmap positive, no gradient at its peak, footprint weight 0
    gu = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in gts[0.0][0].items()}
    bx0 = gu["meta"]["boxes"][0]
    gu["meta"] = dict(gu["meta"], boxes=[dict(bx0, observed=False)] + gu["meta"]["boxes"][1:])
    fp = gu["instance"] == 0
    gu["ignore"] = fp.astype(np.float32)
    gu[bx0["group"]] = np.where(fp, 0.0, gu[bx0["group"]]).astype(np.float32)
    Tu = build_layer_targets([gu], cfg)
    assert float(Tu["w_obj"][0, 0][torch.as_tensor(fp)].abs().max()) == 0.0, "unobserved footprint must be ignored"
    hm_u = torch.zeros(1, 3, X, Y, requires_grad=True)
    lo_u, tt_u = object_head_terms({"hm": hm_u, **{k: torch.zeros(1, c, X, Y) for k, c in (
        ("off", 2), ("size", 2), ("rot", 2), ("rad", 1), ("vel", 2))}}, Tu["boxes"], Tu["w_obj"], grid)
    lo_u["total"].backward()
    hb0 = head_box(bx0)
    ix, iy = int((hb0["cx"] - grid.x_min) // grid.res), int((hb0["cy"] - grid.y_min) // grid.res)
    gi = [g_ for g_, _ in NUSC_GROUPS].index(bx0["group"])
    assert float(tt_u["hm"][0, gi, ix, iy]) < 1.0 and float(hm_u.grad[0, gi, ix, iy].abs()) == 0.0, "unobserved peak"
    # the loss_extras arms: semimetric 0 at p = t for a soft target (soft IoU is not), boundary term ramps in
    soft = torch.full((1, 1, 4, 4), 0.3)
    one = torch.ones_like(soft)
    assert overlap_loss(soft, soft, one, LossConfig(area_overlap="semimetric")).abs().max() < 1e-6
    assert overlap_loss(soft, soft, one, LossConfig()).min() > 0.1
    arm, w1 = LossConfig(line_boundary=True), torch.ones(1, 1, X, Y)
    s0, Ta = leaf_preds(X, 1)[0.0]["sem"], build_layer_targets(gts[0.0], arm)
    _, g0 = map_terms(s0, Ta, spec, {"area": w1, "line": w1, "object": w1}, arm, step_frac=0.0)
    _, g1 = map_terms(s0, Ta, spec, {"area": w1, "line": w1, "object": w1}, arm, step_frac=1.0)
    assert float(g1["lines"]) > float(g0["lines"]), (g0["lines"], g1["lines"])
    # dt weights: every nominal |dt| needs a listed weight; [[0, 1]] alone (the old no-time control) gave dt = +-0.5 the weight 1
    assert [dt_weight(d) for d in (0.0, 0.5, -0.5, 1.0)] == [1.0, 0.5, 0.5, 0.25]
    assert dt_weight(0.5, cfg=LossConfig(dt_weights=((0.0, 1.0), (0.5, 0.0)))) == 0.0
    try:
        dt_weight(0.5, cfg=LossConfig(dt_weights=((0.0, 1.0),)))
        raise AssertionError("an |dt| without a listed weight must be refused")
    except ValueError:
        pass
    print(f"losses selftest ok ({time.time() - t0:.0f} s): soft BCE = analytic {exp:.5f}, tolerant Dice 0 at a 1-cell "
          f"shift / {d3:.2f} at 3 cells, Huber 0 for the exact field, weight-0 cells: zero gradient (all heads), "
          f"doubled map: every term unchanged (max rel. diff {worst:.1e}), Kendall s in [-3, 1], dt weights + warm-up, "
          f"overfit 100 steps: total {t_first:.3f} -> {t_last:.3f}, unweighted sum {r_first:.3f} -> {r_last:.3f}; "
          f"{len(res)} terms; s = " + ", ".join(f"{g} {float(res['s.' + g]):+.2f}" for g in TASK_GROUPS))


if __name__ == "__main__":
    _selftest()
