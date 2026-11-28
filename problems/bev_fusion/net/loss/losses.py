"""Losses of the BEV system (the loss specification; proposal of 2026-10-26).

    per target time dt: GT of t_c + dt (GtGenerator layout) -> per-layer targets + per-cell weights (masks)
      -> per-output losses: areas, lines (+ centre-lines), dense objects (semantic head), object head stage 1,
         motion head, camera / radar auxiliary heads, radar stationary-structure hook
      -> task groups (areas, lines, objects, motion), each term normalised by its own count
    total = sum_g exp(-s_g) L_g + s_g  (Kendall, 1705.07115; or fixed weights)  + fixed-weight auxiliary terms,
    with L_g = sum_dt w(dt) L_g(dt), w(0) = 1, w(+-0.5) = 0.5, w(+-1.0) = 0.25 and a warm-up on the dt != 0 terms.

The six open decisions of 8.4 are options of LossConfig, with the proposal's values as defaults: (1) w_occluded_map,
(2) dense_objects, (3) radar_stationary, (4) use_w_reg (tolerance always; the per-frame weight meta["w_reg"] when the
estimator writes it), (5) dt_weights / warmup_hold / warmup_ramp, (6) task_weighting.
The parts live next to this file: object_loss.py (object targets and head loss), motion_loss.py, radar_loss.py, loss_extras.py; the channel layout is the
network's (layers.py).
The semantic head's channels, their targets and their losses come from the layer table (layers.py: a row = name, kind, level; the kind decides
them); layer_spec lists the channels of the 0.5 m head (SemanticHead(c, len(spec)) gives logits in that order; distance channels are metres,
linear), layer_spec_1m those of the optional 1 m branch (LossConfig.centreline_1m).
Tests: tests/test_losses.py
"""
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bev_fusion.net.model.layers import DIST_MAX, layer_table, pool_targets, source_keys
from bev_fusion.net.model.lift_pd import Grid
from bev_fusion.net.loss.loss_extras import _wcount, _wsum
from bev_fusion.net.model.layers import layer_spec, layer_spec_1m, radar_aux_spec  # noqa: F401
from bev_fusion.net.loss.motion_loss import motion_loss
from bev_fusion.net.loss.object_loss import build_targets, head_loss
from bev_fusion.net.loss.radar_loss import aux_weight, slot_aux_loss  # noqa: F401 (the radar side loss: one of the network's losses)
from bev_fusion.net.prep.targets import NUSC_GROUPS, head_box, sample_targets  # noqa: F401

TASK_GROUPS = ("areas", "lines", "objects", "motion")
EPS = 1e-6


@dataclass
class LossConfig:
    # masks (8.4 table)
    w_occluded_map: float = 0.5        # decision (1): map cells behind a GT box (arm 0 / 1)
    w_unknown: float = 0.5             # unknown at the target time (splat mass < mass_threshold, dt != 0)
    mass_threshold: float = 0.5
    w_interpolated: float = 0.5        # interpolated boxes (12 Hz sweeps); the map stays exact
    use_w_reg: bool = True             # decision (4): per-frame registration weight meta["w_reg"] on lines, if present
    fusion_area_only: bool = False     # D37 / D38 (b), first step; off = every stored run is reproduced. On: the fused heads (areas, lines,
                                       # dense objects, object head, motion) get no loss outside the fusion area = camera x radar cells
                                       # (a cell or an object centre there is neither positive nor negative); the aux heads keep their own area
    # per output
    w_area_iou: float = 0.5
    line_tolerance: int = 1            # cells of the dilated Dice
    huber_delta: float = 0.5           # m
    dist_near: float = 1.0             # m: Huber weight 1 below, dist_w_far above
    dist_w_far: float = 0.25
    dist_max: float = DIST_MAX
    centrelines: bool = False          # the 0.5 m centre-line rows of the layer table (mask + direction); off by default, no stored config sets it
    # plan 2026-11-03: the centre-line row at the 1 m level (layers.py): presence + distance field + direction, targets pooled from the 0.5 m GT (pool_targets);
    # the model's sem_1m output (ModelConfig.centreline_1m must agree). Off = every stored run is reproduced
    centreline_1m: bool = False
    w_direction: float = 0.5
    dense_objects: bool = True         # decision (2): footprint coverage in the semantic head
    w_obj_vel: float = 0.2             # object_head.head_loss velocity weight
    w_obj_mov: float = 1.0             # its moving / static BCE weight (only with model.object_moving, t11)
    max_obj: int = 128
    motion_min_speed: float = 1.0
    motion_w_background: float = 0.1
    w_cam_aux: float = 0.3
    w_radar_aux: float = 0.3
    # decision (3): the proxy target, used when the GT dict carries it. NO GT layer carries it (GtGenerator writes none), so its branch never runs and the radar
    # aux head has one untrained channel; false removes the channel and changes the head layout and the init (open_findings #4: at the next baseline)
    radar_stationary: bool = True
    # lead 2026-11-02, off = every stored run is reproduced bit for bit. On: (#3) the object head and the motion loss apply w_obj as a WEIGHT (w_interpolated,
    # w_unknown, ... reach the focal loss per cell, the regression terms per object and the motion L1 per cell; they only decided keep / ignore before), (#2) an
    # object of unknown velocity (meta vel_known False) trains neither the moving / static logit nor the velocity, (#1) batches.batch_ego raises on a dt other
    # than +-0.5 s (it silently used the 0.5 s gap), the motion loss leaves the cells of a box of unknown velocity (flow_weight 0) out. The weights are
    # RELATIVE: every normaliser clamps at 1, so a batch whose objects are all down-weighted equally is not down-weighted; at dt 0 w_obj is 0 or 1 (fractional
    # only at the ignore edges), the weights act at dt +-0.5 s (w_unknown) and on interpolated labels (w_interpolated); `objhead.*`, `group.objects` and the val
    # total are not comparable between a flag-on and a flag-off arm.
    hygiene_2026_11_02: bool = False
    # time, decision (5)
    dt_weights: tuple = ((0.0, 1.0), (0.5, 0.5), (1.0, 0.25))       # (|dt| nominal, w), nearest |dt| is used
    warmup_hold: float = 0.2           # dt = 0 only for this fraction of the steps
    warmup_ramp: float = 0.2           # then the dt != 0 terms ramp in linearly over this fraction
    # task groups, decision (6)
    task_weighting: str = "kendall"    # or "fixed"
    fixed_weights: tuple = (1.0, 1.0, 1.0, 1.0)
    s_range: tuple = (-3.0, 1.0)


# ------------------------------------------------------------------------------------------------ layer spec


def _t(a, device):
    return torch.as_tensor(np.asarray(a, dtype=np.float32), device=device)


def _stack(gts, key, device):
    """(B, 1, X, Y) for (X, Y) layers, (B, K, X, Y) for (K, X, Y) layers."""
    x = torch.stack([_t(g[key], device) for g in gts])
    return x.unsqueeze(1) if x.dim() == 3 else x


def _stack_many(gts, keys, device):
    """{key: (B, 1 or K, X, Y)} as `_stack`, bit for bit, for all keys at once: the layers of all samples are stacked on the HOST, joined along the
    channel axis and moved with ONE copy; the result is sliced back into views. (A copy per layer per sample was ~1300 of the step's copies.)"""
    keys = list(dict.fromkeys(keys))
    parts = []
    for k in keys:
        a = np.stack([np.asarray(g[k], dtype=np.float32) for g in gts])
        parts.append(a[:, None] if a.ndim == 3 else a)
    big = torch.from_numpy(np.concatenate(parts, 1)).to(device)
    out, c = {}, 0
    for k, p in zip(keys, parts):
        out[k] = big[:, c:c + p.shape[1]]
        c += p.shape[1]
    return out


def build_layer_targets(gts, cfg=None, mass=None, dt=0.0, device="cpu"):
    """gt_bev GT dicts of one target time (a list, one per batch sample, or one dict) -> dict of tensors:
    every layer by its GT key (B, 1 or K, X, Y), 'obj_cells' (instance >= 0), 'boxes' (meta boxes per sample) and the
    per-cell weights (B, 1, X, Y): w_map (areas: undefined 0, occluded w_occluded_map), w_lines (w_map x w_reg),
    w_obj (objects, velocity: (camera and not occluded) or radar; x w_interpolated on interpolated labels),
    w_cam (camera and not occluded), w_radar (radar), w_fusion (camera x radar, D37); all x w_unknown where mass <
    threshold at dt != 0. cfg.fusion_area_only (D38 (b), default off): w_map, w_lines and w_obj are also x w_fusion.
    cfg.centreline_1m (default off): + the pooled 1 m targets 'centreline@1m', 'centreline_dist@1m', 'centreline_dir@1m' and the weight 'w_lines@1m'."""
    cfg = cfg or LossConfig()
    gts = [gts] if isinstance(gts, dict) else list(gts)
    T = {}
    rows = layer_table(cfg.centrelines, True, cfg.centreline_1m)          # the objects are always read: the aux heads' targets
    has_stat = cfg.radar_stationary and all("radar_stationary" in g for g in gts)       # hook: another session builds it
    # GtGenerator: 1 where the velocity is known, 0 on the cells of a box of unknown velocity (velocity 0 there)
    has_flow = cfg.hygiene_2026_11_02 and all("flow_weight" in g for g in gts)
    has_ign = all("ignore" in g for g in gts)
    S = _stack_many(gts, source_keys(rows) + ["velocity", "instance", "camera", "occluded", "radar", "undefined"]
                    + (["radar_stationary"] if has_stat else []) + (["flow_weight"] if has_flow else []) + (["ignore"] if has_ign else []), device)
    for k in source_keys(rows) + ["velocity"]:
        T[k] = S[k]
    if has_stat:
        T["radar_stationary"] = S["radar_stationary"]
    if has_flow:
        T["flow_weight"] = S["flow_weight"]
    T["obj_cells"] = (S["instance"] >= 0).float()
    cam, occ = S["camera"], S["occluded"]
    radar, undef = S["radar"], S["undefined"]
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
    keep = (1 - S["ignore"].clamp(0, 1)) if has_ign else torch.ones_like(cam)
    w_obj = w_obj * keep
    fusion = cam * radar                              # D37: the cells that camera and radar both observe
    if cfg.fusion_area_only:                          # D38 (b): don't care outside it, for the fused heads
        w_map, w_obj = w_map * fusion, w_obj * fusion
    w_reg = torch.tensor([float(g.get("meta", {}).get("w_reg", 1.0)) if cfg.use_w_reg else 1.0 for g in gts]).to(device).view(-1, 1, 1, 1)
    W = {"w_map": w_map, "w_lines": w_map * w_reg, "w_obj": w_obj, "w_cam": w_cam, "w_radar": radar,
         "w_keep": keep, "w_fusion": fusion}
    if mass is not None and abs(float(dt)) > 1e-6:
        m = mass.detach().to(device)
        u = torch.where(m < cfg.mass_threshold, torch.full_like(m, cfg.w_unknown), torch.ones_like(m))
        W = {k: v * u for k, v in W.items()}
    T.update(W)
    T.update(pool_targets(rows, T))                   # the 1 m level (centreline_1m): targets + w_lines@1m from the 0.5 m ones; nothing while it is off
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


def map_terms(logits, T, spec, weights, cfg):
    """Losses of a semantic-type head (main or auxiliary) whose channels follow `spec`. weights: kind group -> (B, 1,
    X, Y) ('area', 'line', 'object', 'stationary'). Returns (terms, groups): per-channel terms and the group sums
    (per-channel terms averaged inside a group, so a group is of order 1)."""
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
        bce, iou = soft_bce(x, t, w), soft_iou_loss(torch.sigmoid(x), t, w)
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
    if "centreline" in idx:                                      # the first centre-line design: presence only, its direction follows
        w = weights["line"]
        dice = tolerant_dice_loss(torch.sigmoid(ch("centreline")), tg("centreline"), w, cfg.line_tolerance)[0]
        terms["centreline_dice"] = dice
        groups["lines"] = groups.get("lines", 0.0) + dice
    if "dir" in idx:                                             # the direction of a polyline row (the centreline mask, or the 1 m row with its distance field)
        w, (name, _, key) = weights["line"], spec[idx["dir"][0]]
        pres = next(k for n, kd, k in spec if n == name.rsplit("_", 1)[0] and kd in ("line", "centreline"))
        dl = direction_l1(ch("dir"), T[key], T[pres], w)
        terms[key] = dl
        groups["lines"] = groups.get("lines", 0.0) + cfg.w_direction * dl
    if "object" in idx:
        x, t, w = ch("object"), tg("object"), weights["object"]
        bce, iou = soft_bce(x, t, w), soft_iou_loss(torch.sigmoid(x), t, w)
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


def object_head_terms(out, boxes_batch, w_obj, grid, groups=NUSC_GROUPS, w_vel=0.2, max_obj=128, w_mov=1.0, hygiene=False):
    """Stage 1 via object_head.head_loss. A box is supervised if it is observed (gt_bev meta 'observed': any lidar /
    radar point) and its centre cell has w_obj > 0; the others are ignored (neither positive nor negative: their peak
    region and every cell of w_obj = 0 get a constant logit, so no loss gradient). Returns (head_loss dict, targets).
    hygiene (LossConfig.hygiene_2026_11_02): w_obj is also the WEIGHT of the supervised ones (the focal loss per cell, every regression term per object by its
    centre cell) and an object of
    unknown velocity (vel_known False) trains neither the moving / static logit nor the velocity."""
    X, Y = w_obj.shape[-2:]
    # ONE device-to-host read per call (float(w_obj[...]) per box was one GPU sync per box: ~270 per step)
    w_host = w_obj[:, 0].detach().cpu()
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
            wv = float(w_host[b, ix, iy])
            # the weight of this object and whether its velocity is known travel with the target box
            if hygiene:
                hb = dict(hb, loss_w=wv, vel_known=bool(bx.get("vel_known", True)))
            (k if bx.get("observed", True) and wv > 0 else d).append(hb)
        keep.append(k)
        drop.append(d)
    dev = out["hm"].device
    t = build_targets(keep, grid, groups, max_obj)
    ign = (w_obj <= 0).expand_as(out["hm"])
    if any(drop):
        ign = ign | ((build_targets(drop, grid, groups, max_obj)["hm"].to(dev) > 0) & (t["hm"].to(dev) == 0))
    hm = torch.where(ign, torch.full_like(out["hm"], -1e4), out["hm"])
    return head_loss(dict(out, hm=hm), t, w_vel, w_mov=w_mov, cell_w=w_obj if hygiene else None), t


def motion_term(v_hat, T, cfg):
    """motion_loss on the cells where objects / velocity are supervised (w_obj > 0), normalised by their counts. LossConfig.hygiene_2026_11_02: the cells of a
    box of unknown velocity (flow_weight 0, GT velocity 0) are out of it, as the evaluation leaves them out, and w_obj weights the rest (it is 0 or 1 at dt 0
    but at the ignore edges: the weight acts at dt +-0.5 s)."""
    valid = T["w_obj"][:, 0] > 0
    if cfg.hygiene_2026_11_02 and "flow_weight" in T:
        valid = valid & (T["flow_weight"][:, 0] > 0)

    def sel(x):
        return x.permute(0, 2, 3, 1)[valid].t().reshape(1, x.shape[1], -1, 1)
    return motion_loss(sel(v_hat), sel(T["velocity"]), sel(T["obj_cells"]), cfg.motion_min_speed,
                           cfg.motion_w_background, weight=sel(T["w_obj"]) if cfg.hygiene_2026_11_02 else None)


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
    preds: {dt: {'sem': (B, len(spec), X, Y) logits, 'sem_1m': (B, len(spec_1m), X / 2, Y / 2) logits (centreline_1m only; its terms are
            'sem1m.*' and join the lines group), 'obj': object head maps, 'v_hat': (B, 2, X, Y) motion field
            (supervised at dt = 0 against the GT of t_c), 'mass': (B, 1, X, Y) splat mass at dt,
            'cam_aux': (B, len(cam_spec), X, Y), 'radar_aux': (B, len(radar_spec), X, Y)}; every entry optional
    gts: {dt: [gt_bev GT dict per sample]} (sample_targets builds it from training samples)
    Returns a dict: every term as '<name>@<dt>', 'group.<g>' (dt-weighted), 's.<g>', 'aux', 'total'."""

    def __init__(self, cfg=None, grid=None, groups=NUSC_GROUPS):
        super().__init__()
        self.cfg, self.grid, self.groups = cfg or LossConfig(), grid or Grid(), groups
        c = self.cfg
        self.spec = layer_spec(c.centrelines, c.dense_objects)
        self.spec_1m = layer_spec_1m(c.centreline_1m)
        self.cam_spec = layer_spec(False, True)
        self.radar_spec = radar_aux_spec(c.radar_stationary)
        self.task = TaskWeights(TASK_GROUPS, c.task_weighting, c.fixed_weights, c.s_range)

    def terms_at(self, p, gts, dt):
        """(terms, task-group losses, fixed-weight auxiliary losses) of one target time."""
        cfg = self.cfg
        dev = next(v for v in p.values() if torch.is_tensor(v)) if "obj" not in p else p["obj"]["hm"]
        T = build_layer_targets(gts, cfg, p.get("mass"), dt, dev.device)
        terms, groups, fixed = {}, {}, {}

        def add(g, v):
            groups[g] = groups.get(g, 0.0) + v
        if "sem" in p:
            w = {"area": T["w_map"], "line": T["w_lines"], "object": T["w_obj"]}
            t, g = map_terms(p["sem"], T, self.spec, w, cfg)
            terms.update(("sem." + k, v) for k, v in t.items())
            for k, v in g.items():
                add(k, v)
        # the layout flag lives in two configs (LossConfig, ModelConfig): a mismatch must not pass silently
        if bool(self.spec_1m) != ("sem_1m" in p):
            raise ValueError(f"centreline_1m: the loss has {len(self.spec_1m)} 1 m channels but the prediction {'has' if 'sem_1m' in p else 'has no'} 'sem_1m'")
        if self.spec_1m:
            t, g = map_terms(p["sem_1m"], T, self.spec_1m, {"line": T["w_lines@1m"]}, cfg)
            terms.update(("sem1m." + k, v) for k, v in t.items())
            for k, v in g.items():
                add(k, v)
        tt = None
        if "obj" in p:
            lo, tt = object_head_terms(p["obj"], T["boxes"], T["w_obj"], self.grid, self.groups, cfg.w_obj_vel,
                                       cfg.max_obj, cfg.w_obj_mov, hygiene=cfg.hygiene_2026_11_02)
            terms.update(("objhead." + k, v) for k, v in lo.items())
            add("objects", lo["total"])
        if "v_hat" in p and abs(float(dt)) < 1e-6:
            terms["motion"] = motion_term(p["v_hat"], T, cfg)
            lv = [motion_term(v, T, cfg) for v in (p.get("v_levels") or {}).values()]
            for i, t in enumerate(lv):                           # t15: the motion block's coarser levels (deep supervision, as in t12)
                terms[f"motion_l{i}"] = t
            add("motion", (terms["motion"] + sum(lv)) / (1 + len(lv)))   # the mean over the levels: the group keeps the weight of one head
        if "cam_aux" in p:
            wc = T["w_cam"]
            t, g = map_terms(p["cam_aux"], T, self.cam_spec, {"area": wc, "line": wc, "object": wc * T["w_keep"]}, cfg)
            terms.update(("cam_aux." + k, v) for k, v in t.items())
            fixed["cam_aux"] = cfg.w_cam_aux * sum(g.values())
        if "radar_aux" in p:
            wr = T["w_radar"]
            t, g = map_terms(p["radar_aux"], T, self.radar_spec, {"object": wr * T["w_keep"], "stationary": wr}, cfg)
            terms.update(("radar_aux." + k, v) for k, v in t.items())
            fixed["radar_aux"] = cfg.w_radar_aux * sum(g.values())
        return terms, groups, fixed

    def forward(self, preds, gts, step_frac=1.0):
        out, group_tot, aux = {}, {}, 0.0
        for dt, p in preds.items():
            w = dt_weight(dt, step_frac, self.cfg)
            if w <= 0:
                continue
            terms, groups, fixed = self.terms_at(p, gts[dt], dt)
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
