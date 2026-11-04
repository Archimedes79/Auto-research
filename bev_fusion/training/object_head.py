"""Object head, stage 1 (plan.md section 8.3, S5a): CenterPoint-style centre points on the BEV grid.

Pixel-based: every cell is classified as an object centre per class group (heatmap); at the centre cell the head
regresses the sub-cell offset, the velocity and, by kind of group, an oriented box (log length, log width, yaw as
sin, cos) or a circle (log radius). Decoding: local maxima of the heatmap (3 x 3 max-pool), top-K, score threshold.
Training targets: Gaussian peaks around the GT centres with CenterPoint's radius rule (min overlap 0.1, at least 2
cells); losses: focal loss on the heatmap (CenterNet), L1 on the regressions at the GT centre cells.
Basis: CenterNet "Objects as Points" (1904.07850, from memory), CenterPoint 2006.11275 (lit; MIT code
`det3d/models/bbox_heads/center_head.py` - re-implemented here, nothing copied).
Boxes are dicts: cx, cy (m, vehicle frame), length, width (m), yaw (rad), radius (m, circles), vx, vy (m/s), group.
Run the self-test: python training/object_head.py
"""
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from . import launch_saving as LS
except ImportError:                                   # run as a script
    import launch_saving as LS

GROUPS = (("car", "box"), ("truck", "box"), ("pedestrian", "circle"))     # (name, kind); synthetic-world classes


def _branch(cin, hidden, cout, bias=0.0):
    b = nn.Sequential(nn.Conv2d(cin, hidden, 3, padding=1), nn.ReLU(), nn.Conv2d(hidden, cout, 1))
    nn.init.constant_(b[2].bias, bias)
    return b


DIR_OFFSET = -math.pi / 4      # direction classes split the heading circle at -45 deg / 135 deg (diagonal: rare on roads)
MOVING_SPEED = 0.5             # m/s: target of the mov logit, = evaluation/metrics.py MOVING_SPEED (the AVE's moving / static split)


def direction_class(yaw):
    """1 if the heading lies in [DIR_OFFSET, DIR_OFFSET + pi) (mod 2 pi), else 0 (SECOND's direction target)."""
    return float(((yaw - DIR_OFFSET) % (2 * math.pi)) < math.pi)


class ObjectHead(nn.Module):
    """BEV features (B, C, X, Y) -> dict of maps: hm (B, G, X, Y) logits, off (2), size (2: log l, log w),
    rot (2: sin, cos), rad (1: log r), vel (2). direction=True (t1 round 1, 2026-10-26): also dir (1 logit); the rot
    loss is then sign-free (the box axis) and the front / back half comes from dir, as SECOND (Yan et al. 2018) /
    PointPillars: the plain L1 on (sin, cos) did not learn front vs back on nuScenes in 2000 steps (rot loss stayed at
    the level of random directions while the axis was right). moving=True (t11 hurdle, 2026-11-01): also mov (1 logit,
    moving / static); the velocity is then regressed on moving centres only and decode zeroes it where mov says static:
    the L1 over all centres, 64 % of them static, had its minimiser (the median) at 0 m/s."""

    def __init__(self, cin, groups=GROUPS, hidden=64, direction=False, moving=False):
        super().__init__()
        self.groups = groups
        self.shared = nn.Sequential(nn.Conv2d(cin, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU())
        self.hm = _branch(hidden, hidden, len(groups), bias=-2.19)        # sigmoid(-2.19) = 0.1 (CenterNet init)
        self.off, self.size = _branch(hidden, hidden, 2), _branch(hidden, hidden, 2)
        self.rot, self.rad, self.vel = _branch(hidden, hidden, 2), _branch(hidden, hidden, 1), _branch(hidden, hidden, 2)
        self.names = ("hm", "off", "size", "rot", "rad", "vel") + (("dir",) if direction else ()) + (("mov",) if moving else ())
        if direction:
            self.dir = _branch(hidden, hidden, 1)
        if moving:
            self.mov = _branch(hidden, hidden, 1, bias=-1.39)             # sigmoid(-1.39) = 0.2: prior share of moving centres

    def forward(self, feat):
        x = self.shared(feat)
        return {k: getattr(self, k)(x) for k in self.names}


# ------------------------------------------------------------------------------------------------ targets
def gaussian_radius(h, w, min_overlap=0.1):
    """CornerNet / CenterPoint radius (in cells) of the Gaussian peak for a box of h x w cells."""
    b1, c1 = h + w, w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(max(0.0, b1 * b1 - 4 * c1))) / 2
    b2, c2 = 2 * (h + w), (1 - min_overlap) * w * h
    r2 = (b2 + math.sqrt(max(0.0, b2 * b2 - 16 * c2))) / 2
    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (h + w), (min_overlap - 1) * w * h
    r3 = (b3 + math.sqrt(max(0.0, b3 * b3 - 4 * a3 * c3))) / 2
    return min(r1, r2, r3)


def build_targets(boxes_batch, grid, groups=GROUPS, max_obj=64, min_radius=2):
    """boxes_batch: list (B) of lists of box dicts -> dict of target tensors (hm, ind, mask, group, off, size, rot,
    rad, vel, is_box, dir). `mask` is the box's `loss_w` (default 1); when any box carries `vel_known` (LossConfig.hygiene_2026_11_02) the dict also
    has `known` (1 where the velocity of the object is known). The per-box fields are made for a whole frame at once in numpy and handed over with one
    torch.from_numpy per field (the version before made a torch.tensor per field per box: 3.6 x slower); the Gaussian windows of a frame are
    exponentiated in ONE torch.exp call, and torch.exp stays: numpy's exp differs from it in the last bit for ~40 % of the values, so the heatmaps are
    bit-identical to the old version (tests/test_host_targets.py checks all tensors with torch.equal against a copy of the old function, including
    loss_w / vel_known, the cap, unknown groups outside the grid and the inputs that used to raise)."""
    B, G, X, Y = len(boxes_batch), len(groups), grid.nx, grid.ny
    gidx = {name: i for i, (name, _) in enumerate(groups)}
    box_kind = np.array([k == "box" for _, k in groups])
    has_known = any("vel_known" in bx for boxes in boxes_batch for bx in boxes)
    hm = np.zeros((B, G, X, Y), np.float32)
    ind, grp = np.zeros((B, max_obj), np.int64), np.zeros((B, max_obj), np.int64)
    mask, is_box = np.zeros((B, max_obj), np.float32), np.zeros((B, max_obj), np.float32)
    known = np.ones((B, max_obj), np.float32) if has_known else None
    off, size, rot, vel = (np.zeros((B, max_obj, 2), np.float32) for _ in range(4))
    rad, dcl = np.zeros((B, max_obj, 1), np.float32), np.zeros((B, max_obj, 1), np.float32)
    f64 = lambda xs: np.array(xs, dtype=np.float64)                                   # noqa: E731 - python float semantics, also for numpy scalar inputs
    for b, boxes in enumerate(boxes_batch):
        if not boxes:
            continue
        fx = (f64([bx["cx"] for bx in boxes]) - grid.x_min) / grid.res - 0.5          # continuous cell index of the centre
        fy = (f64([bx["cy"] for bx in boxes]) - grid.y_min) / grid.res - 0.5
        if not (np.isfinite(fx).all() and np.isfinite(fy).all()):
            raise ValueError("build_targets: a box centre is not finite")           # the old int(math.floor(nan)) raised as well
        ix, iy = np.floor(fx + 0.5).astype(np.int64), np.floor(fy + 0.5).astype(np.int64)
        sel = np.nonzero((ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y))[0][:max_obj]
        n = len(sel)
        if not n:
            continue
        fx, fy, ix, iy = fx[sel], fy[sel], ix[sel], iy[sel]
        bb = [boxes[i] for i in sel]
        g = np.array([gidx[bx["group"]] for bx in bb])                             # only for the boxes that are kept (an unknown group outside the grid is skipped)
        isb = box_kind[g]
        length = f64([bx["length"] if k else 1.0 for bx, k in zip(bb, isb)])
        width = f64([bx["width"] if k else 1.0 for bx, k in zip(bb, isb)])
        radius = f64([1.0 if k else bx["radius"] for bx, k in zip(bb, isb)])
        yaw = f64([bx["yaw"] if k else 0.0 for bx, k in zip(bb, isb)])
        if np.any(np.where(isb, (length <= 0) | (width <= 0), radius <= 0)):
            raise ValueError("math domain error: a box length / width / radius <= 0")     # the old math.log raised
        r = np.array([gaussian_radius(ln / grid.res, w / grid.res) if k else gaussian_radius(2 * rd / grid.res, 2 * rd / grid.res)
                      for ln, w, rd, k in zip(length.tolist(), width.tolist(), radius.tolist(), isb)])
        r = np.maximum(min_radius, r.astype(np.int64))
        sigma = (2 * r + 1) / 6
        args, wins = [], []
        for cx_, cy_, r_, s_ in zip(ix.tolist(), iy.tolist(), r.tolist(), sigma.tolist()):      # python numbers: float32 stays float32 (numpy 2 would promote an int64 scalar)
            x0, x1, y0, y1 = max(0, cx_ - r_), min(X, cx_ + r_ + 1), max(0, cy_ - r_), min(Y, cy_ + r_ + 1)
            gx = np.arange(x0, x1, dtype=np.float32)[:, None]
            gy = np.arange(y0, y1, dtype=np.float32)[None, :]
            args.append((-((gx - cx_) ** 2 + (gy - cy_) ** 2) / np.float32(2 * s_ * s_)).reshape(-1))
            wins.append((x0, x1, y0, y1))
        peaks = torch.exp(torch.from_numpy(np.concatenate(args))).numpy()
        c = 0
        for k, (x0, x1, y0, y1) in enumerate(wins):
            m = (x1 - x0) * (y1 - y0)
            win = hm[b, g[k], x0:x1, y0:y1]
            np.maximum(win, peaks[c:c + m].reshape(x1 - x0, y1 - y0), out=win)
            c += m
        ind[b, :n], grp[b, :n] = ix * Y + iy, g
        mask[b, :n] = [bx.get("loss_w", 1.0) for bx in bb]
        if has_known:
            known[b, :n] = [1.0 if bx.get("vel_known", True) else 0.0 for bx in bb]
        off[b, :n] = np.stack([fx - ix, fy - iy], 1)
        vel[b, :n] = np.stack([f64([bx["vx"] for bx in bb]), f64([bx["vy"] for bx in bb])], 1)
        size[b, :n] = np.where(isb[:, None], np.log(np.stack([length, width], 1)), 0.0)
        rot[b, :n] = np.where(isb[:, None], np.stack([np.sin(yaw), np.cos(yaw)], 1), 0.0)
        dcl[b, :n, 0] = np.where(isb, ((yaw - DIR_OFFSET) % (2 * math.pi) < math.pi), 0.0)
        rad[b, :n, 0] = np.where(isb, 0.0, np.log(radius))
        is_box[b, :n] = isb
    out = dict(hm=hm, ind=ind, mask=mask, group=grp, off=off, size=size, rot=rot, rad=rad, vel=vel, is_box=is_box, dir=dcl)
    if has_known:
        out["known"] = known
    return {k: torch.from_numpy(v) for k, v in out.items()}


# ------------------------------------------------------------------------------------------------ losses
def focal_loss(logits, target, weight=None):
    """CenterNet focal loss (alpha 2, beta 4), normalised by the number of peaks. weight (B, 1, X, Y) or None: a per-cell weight (LossConfig.hygiene_2026_11_02); the sums are then weighted and
    the normaliser is the weighted number of peaks. None = the plain loss, bit for bit."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg = 1 - pos
    pos_l = -torch.log(p) * (1 - p) ** 2 * pos
    neg_l = -torch.log(1 - p) * p ** 2 * (1 - target) ** 4 * neg
    if weight is not None:
        w = weight.to(p.dtype).expand_as(p)
        return ((pos_l * w).sum() + (neg_l * w).sum()) / (pos * w).sum().clamp(min=1)
    return (pos_l.sum() + neg_l.sum()) / pos.sum().clamp(min=1)


def _gather(m, ind):
    """m (B, C, X, Y), ind (B, M) flat cell index -> (B, M, C)."""
    B, C = m.shape[:2]
    return m.reshape(B, C, -1).gather(2, ind.unsqueeze(1).expand(B, C, ind.shape[1])).permute(0, 2, 1)


def head_loss(out, t, w_vel=0.2, w_dir=0.2, w_mov=1.0, cell_w=None):
    """dict of loss terms and their sum 'total'. With out['dir'] (ObjectHead(direction=True)): rot is the sign-free axis
    loss min(L1(r, t), L1(r, -t)) and 'dir' the BCE of the direction class (weight w_dir, SECOND's 0.2). With out['mov']
    (ObjectHead(moving=True)): 'mov' the BCE of moving (|vel| > MOVING_SPEED) at every centre (weight w_mov), and 'vel'
    the L1 on the moving centres only (a hurdle model: the static ones no longer pull the regression to 0).
    LossConfig.hygiene_2026_11_02 (all three absent = the plain loss, bit for bit): cell_w (B, 1, X, Y) weights the focal loss per cell; t["mask"] may hold a weight per object (build_targets: the
    `loss_w` of a box dict), which then weights every regression term; t["known"] (1 where the velocity of the object is known) takes the objects of unknown velocity out of the moving / static
    BCE and of the velocity L1 (their velocity target is 0, which would read as "static")."""
    dev = out["hm"].device
    t = {k: LS.to_dev(v, dev) for k, v in t.items()}         # D54: with the sync switch on, from pinned memory without a wait for the GPU; else v.to(dev)
    m = t["mask"].unsqueeze(-1)
    mb = (t["mask"] * t["is_box"]).unsqueeze(-1)
    mc = (t["mask"] * (1 - t["is_box"])).unsqueeze(-1)
    mk = m * t["known"].unsqueeze(-1) if "known" in t else m
    mv = mk * (t["vel"].norm(dim=-1, keepdim=True) > MOVING_SPEED).float() if "mov" in out else mk
    cell_w = None if cell_w is None else cell_w.to(dev)

    def l1(name, mask):
        return (F.l1_loss(_gather(out[name], t["ind"]), t[name], reduction="none") * mask).sum() / mask.sum().clamp(min=1)
    rot = _gather(out["rot"], t["ind"])
    rot = rot / rot.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    e_rot = F.l1_loss(rot, t["rot"], reduction="none").sum(-1, keepdim=True)
    if "dir" in out:
        e_rot = torch.minimum(e_rot, F.l1_loss(rot, -t["rot"], reduction="none").sum(-1, keepdim=True))
    losses = {"hm": focal_loss(out["hm"], t["hm"], cell_w), "off": l1("off", m), "size": l1("size", mb),
              "rot": (e_rot * mb).sum() / mb.sum().clamp(min=1),
              "rad": l1("rad", mc), "vel": w_vel * l1("vel", mv)}
    if "dir" in out:
        bce = F.binary_cross_entropy_with_logits(_gather(out["dir"], t["ind"]), t["dir"], reduction="none")
        losses["dir"] = w_dir * (bce * mb).sum() / mb.sum().clamp(min=1)
    if "mov" in out:
        bce = F.binary_cross_entropy_with_logits(_gather(out["mov"], t["ind"]), (mv > 0).float(), reduction="none")
        losses["mov"] = w_mov * (bce * mk).sum() / mk.sum().clamp(min=1)
    losses["total"] = sum(losses.values())
    return losses


# ------------------------------------------------------------------------------------------------ decoding
_boxes_from_maps = None


@torch.no_grad()
def decode(out, grid, groups=GROUPS, k=50, threshold=0.2, from_probabilities=False):
    """out maps -> list (B) of box dicts with 'score': the local maxima of the heatmap (3 x 3 max-pool), top-k, score threshold. With
    out['mov'] the velocity is 0 where sigmoid(mov) < 0.5 (static: the AVE-optimal answer of a two-point mixture; inference only).
    There is ONE extractor, extract.boxes_from_maps (vectorised, only the K peak cells are read); this is its name in the object head's
    module. extract.py's self-test checks it box by box against the plain per-box loop. from_probabilities: hm already in [0, 1]."""
    global _boxes_from_maps
    if _boxes_from_maps is None:                           # imported on first use: extract imports this module (DIR_OFFSET)
        try:
            from .extract import boxes_from_maps
        except ImportError:
            from extract import boxes_from_maps
        _boxes_from_maps = boxes_from_maps
    return _boxes_from_maps(out, grid, groups, k, threshold, from_probabilities)


class TimeObjectHead(ObjectHead):
    """The head as a function of time (Mirko, 2026-10-26; plan 8.3 S5b, learned arm): the same head, its shared
    features modulated per channel by the time offset (FiLM: scale and shift from an MLP on dt, |dt|, sign(dt));
    forward(feat, dt) predicts the boxes at t + dt in the ego frame of t + dt. The FiLM starts as the identity, so
    dt = 0 behaves like ObjectHead; trained with several dt per scene against the GT boxes at t + dt."""

    def __init__(self, cin, groups=GROUPS, hidden=64):
        super().__init__(cin, groups, hidden)
        self.film = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 2 * hidden))
        nn.init.zeros_(self.film[2].weight)
        nn.init.zeros_(self.film[2].bias)

    def forward(self, feat, dt):
        dt = torch.as_tensor(dt, dtype=feat.dtype, device=feat.device).reshape(-1).expand(feat.shape[0])
        e = torch.stack([dt, dt.abs(), torch.sign(dt)], -1)
        gamma, beta = self.film(e).chunk(2, -1)
        x = self.shared(feat)
        x = x * (1 + gamma[..., None, None]) + beta[..., None, None]
        return {k: getattr(self, k)(x) for k in ("hm", "off", "size", "rot", "rad", "vel")}


# ------------------------------------------------------------------------------------------------ S5b: boxes in time
def propagate_box_tensors(centre, yaw, vel, steps, delta=None, dyaw=None):
    """S5b (plan 8.3): move boxes to another time with the splat's position function (training/splat.py).
    centre (B, N, 2), yaw (B, N) or None (circles), vel (B, N, 2) box velocity in the source axes; steps =
    [(dt_i, v_i, omega_i), ...] ego steps from the source to the target time (any sign of dt); delta (B, N, 2), dyaw
    (B, N): an optional learned correction. Returns centre', yaw', vel' in the target frame; size / radius unchanged.
    Differentiable in every input (the loss on the propagated box trains the velocity head and the correction)."""
    try:
        from .splat import chain_positions
    except ImportError:
        from splat import chain_positions
    c = chain_positions(centre, steps, vel, delta)
    th = sum(float(s[0]) * float(s[2]) for s in steps)
    co, si = math.cos(th), math.sin(th)
    v2 = torch.stack([co * vel[..., 0] + si * vel[..., 1], -si * vel[..., 0] + co * vel[..., 1]], -1)
    y2 = None if yaw is None else yaw - th + (dyaw if dyaw is not None else 0.0)
    return c, y2, v2


def propagate_boxes(boxes, steps):
    """The same for decoded box dicts (constant velocity, no correction): list of dicts -> list of dicts at the target
    time."""
    if not boxes:
        return []
    ctr = torch.tensor([[[b["cx"], b["cy"]] for b in boxes]], dtype=torch.float64)
    vel = torch.tensor([[[b["vx"], b["vy"]] for b in boxes]], dtype=torch.float64)
    yaw = torch.tensor([[b.get("yaw", 0.0) for b in boxes]], dtype=torch.float64)
    c, y, v = propagate_box_tensors(ctr, yaw, vel, steps)
    out = []
    for i, b in enumerate(boxes):
        nb = dict(b, cx=c[0, i, 0].item(), cy=c[0, i, 1].item(), vx=v[0, i, 0].item(), vy=v[0, i, 1].item())
        if "yaw" in b:
            nb["yaw"] = y[0, i].item()
        out.append(nb)
    return out


class BoxCorrection(nn.Module):
    """Bounded learned correction of a propagated box (S5b): (delta x, delta y, delta yaw) from the box's centre feature
    and the ego / time inputs (v, omega, dt, v dt, omega dt) and the box velocity; zero start, |delta| <= d_max m,
    |delta yaw| <= yaw_max rad. For objects that accelerate or turn (constant velocity is the model without it)."""

    def __init__(self, cin, hidden=64, d_max=1.0, yaw_max=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cin + 7, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 3))
        nn.init.zeros_(self.net[4].weight)
        nn.init.zeros_(self.net[4].bias)
        self.d_max, self.yaw_max = d_max, yaw_max

    def forward(self, feat, vel, dt, v, omega):
        """feat (B, N, C) centre features, vel (B, N, 2); dt, v, omega scalars -> delta (B, N, 2), dyaw (B, N)."""
        B, N = feat.shape[:2]
        e = torch.tensor([v, omega, dt, v * dt, omega * dt], dtype=feat.dtype, device=feat.device).expand(B, N, 5)
        raw = self.net(torch.cat([feat, vel, e], -1))
        d = self.d_max * torch.tanh(raw[..., :2] / self.d_max)
        return d, self.yaw_max * torch.tanh(raw[..., 2] / self.yaw_max)


def targets_as_output(t, grid, direction=False):
    """Perfect head output from targets (self-test): hm as probabilities, regressions written at the centre cells;
    direction: also 'dir' as a logit (+-10)."""
    B, G, X, Y = t["hm"].shape
    out = {"hm": t["hm"].clone()}
    names = (("off", 2), ("size", 2), ("rot", 2), ("rad", 1), ("vel", 2)) + ((("dir", 1),) if direction else ())
    for name, c in names:
        m = torch.zeros(B, c, X * Y)
        for b in range(B):
            for n in range(t["ind"].shape[1]):
                if t["mask"][b, n] > 0:
                    m[b, :, t["ind"][b, n]] = t[name][b, n]
        out[name] = m.view(B, c, X, Y)
    if direction:
        out["dir"] = 20.0 * out["dir"] - 10.0
    return out


# ------------------------------------------------------------------------------------------------ S5c: box refinement
# Stage 2 (plan 8.3 S5c): per candidate box 4 sampling points, initialised at the centres of the 4 box faces (front,
# left, rear, right; circles: on the circle at 0 / 90 / 180 / 270 deg), moved by learned bounded offsets in the box
# frame (zero start; deformable=False = CenterPoint's fixed centre + 4 face points, the control); bilinear samples of
# the BEV features there and at the centre -> MLP -> residuals in the box frame and an IoU-quality logit.
# Box tensor (B, N, 7): cx, cy, a, b, yaw, vx, vy; a, b = log length, log width (boxes) or log r, log r (circles, yaw 0).
BOX_DIM = 7


def boxes_to_tensor(box_lists, groups=GROUPS, device=None):
    """list (B) of lists of box dicts -> box (B, N, 7), group (B, N) long, valid (B, N) float (padding 0)."""
    gidx = {name: i for i, (name, _) in enumerate(groups)}
    B, N = len(box_lists), max(1, max((len(b) for b in box_lists), default=0))
    box, group, valid = torch.zeros(B, N, BOX_DIM), torch.zeros(B, N, dtype=torch.long), torch.zeros(B, N)
    for i, boxes in enumerate(box_lists):
        for j, bx in enumerate(boxes):
            g = gidx[bx["group"]]
            if groups[g][1] == "circle":
                a = b = math.log(bx["radius"])
                yaw = 0.0
            else:
                a, b, yaw = math.log(bx["length"]), math.log(bx["width"]), bx["yaw"]
            box[i, j] = torch.tensor([bx["cx"], bx["cy"], a, b, yaw, bx["vx"], bx["vy"]])
            group[i, j], valid[i, j] = g, 1.0
    return box.to(device), group.to(device), valid.to(device)


def tensor_to_boxes(box, group, valid, groups=GROUPS, like=None):
    """Inverse of boxes_to_tensor -> list (B) of lists of box dicts; like: the source lists, whose other keys (score)
    are copied."""
    box, group, valid = box.detach().cpu().tolist(), group.cpu().tolist(), valid.cpu().tolist()
    res = []
    for i in range(len(box)):
        out = []
        for j in range(len(box[i])):
            if valid[i][j] <= 0:
                continue
            name, kind = groups[group[i][j]]
            cx, cy, a, b, yaw, vx, vy = box[i][j]
            bx = dict(like[i][j]) if like is not None else {}
            bx.update(group=name, cx=cx, cy=cy, vx=vx, vy=vy)
            if kind == "box":
                bx.update(length=math.exp(a), width=math.exp(b), yaw=yaw)
            else:
                bx["radius"] = math.exp(a)
            out.append(bx)
        res.append(out)
    return res


def _rot(yaw, v):
    """rotate 2-vectors v (..., 2) by yaw (...)."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)


def sample_bev(feat, pts, grid):
    """Bilinear samples of feat (B, C, X, Y) at pts (B, N, K, 2) in metres (x forward, y left; cell i has its centre
    at x_min + (i + 0.5) res, as lift_pd.Grid) -> (B, N, K, C); zero outside the grid."""
    u = 2 * (pts[..., 1] - grid.y_min) / (grid.ny * grid.res) - 1        # grid_sample: [..., 0] indexes the last dim
    v = 2 * (pts[..., 0] - grid.x_min) / (grid.nx * grid.res) - 1
    s = F.grid_sample(feat, torch.stack([u, v], -1).to(feat.dtype), mode="bilinear", padding_mode="zeros",
                      align_corners=False)
    return s.permute(0, 2, 3, 1)


def apply_residual(box, delta, circle):
    """box (B, N, 7) + residuals delta (B, N, 7: dx, dy in the box frame, da, db, dyaw, dvx, dvy in the box frame);
    circle (B, N) bool (b follows a, yaw stays)."""
    c = box[..., :2] + _rot(box[..., 4], delta[..., :2])
    a = box[..., 2] + delta[..., 2]
    b = torch.where(circle, a, box[..., 3] + delta[..., 3])
    yaw = torch.where(circle, box[..., 4], box[..., 4] + delta[..., 4])
    v = box[..., 5:7] + _rot(box[..., 4], delta[..., 5:7])
    return torch.cat([c, a[..., None], b[..., None], yaw[..., None], v], -1)


def residual_targets(box, gt, circle):
    """The residuals that move box onto gt (inverse of apply_residual); yaw modulo pi (a footprint is symmetric under
    a half turn; the heading direction is left to stage 1 and the velocity); circles: db = dyaw = 0."""
    d = _rot(-box[..., 4], gt[..., :2] - box[..., :2])
    da = gt[..., 2] - box[..., 2]
    db = torch.where(circle, torch.zeros_like(da), gt[..., 3] - box[..., 3])
    dyaw = torch.remainder(gt[..., 4] - box[..., 4] + math.pi / 2, math.pi) - math.pi / 2
    dyaw = torch.where(circle, torch.zeros_like(dyaw), dyaw)
    dv = _rot(-box[..., 4], gt[..., 5:7] - box[..., 5:7])
    return torch.cat([d, da[..., None], db[..., None], dyaw[..., None], dv], -1)


class BoxRefiner(nn.Module):
    """S5c: iterative box refinement (n_iter loops, shared weights, the box detached between loops). Inputs of the MLP
    per candidate: [centre feature; 4 point features; the 4 points relative to the centre in the box frame (m); the box
    velocity in the box frame / 10; group one-hot; dt embedding (dt, |dt|, sign) if use_dt; absolute x, y / 50 only if
    use_abs_xy (ablation arm)]. Outputs: residuals (apply_residual) and an IoU-quality logit of the input box. Offsets
    of the points (deformable=True): from the centre feature, |offset| <= d_max m per axis, zero start."""

    def __init__(self, cin, n_iter=1, use_abs_xy=False, deformable=True, groups=GROUPS, hidden=128, d_max=1.0,
                 use_dt=False):
        super().__init__()
        self.groups, self.n_iter, self.use_abs_xy, self.deformable = groups, n_iter, use_abs_xy, deformable
        self.d_max, self.use_dt = d_max, use_dt
        self.register_buffer("circle_of_group", torch.tensor([k == "circle" for _, k in groups]), persistent=False)
        if deformable:
            self.offset = nn.Sequential(nn.Linear(cin, hidden), nn.ReLU(), nn.Linear(hidden, 8))
            nn.init.zeros_(self.offset[2].weight)
            nn.init.zeros_(self.offset[2].bias)
        nin = 5 * cin + 8 + 2 + len(groups) + (3 if use_dt else 0) + (2 if use_abs_xy else 0)
        self.mlp = nn.Sequential(nn.Linear(nin, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, BOX_DIM + 1))
        nn.init.zeros_(self.mlp[4].weight)
        nn.init.zeros_(self.mlp[4].bias)

    def points(self, box, group, centre_feat=None):
        """Sampling points of the boxes: rel (B, N, 4, 2) in the box frame and world (B, N, 4, 2) in metres; order
        front, left, rear, right face (circles: 0, 90, 180, 270 deg)."""
        circle = self.circle_of_group[group]
        ea, eb = box[..., 2].exp(), box[..., 3].exp()
        hl, hw = torch.where(circle, ea, ea / 2), torch.where(circle, ea, eb / 2)
        z = torch.zeros_like(hl)
        rel = torch.stack([torch.stack(p, -1) for p in ((hl, z), (z, hw), (-hl, z), (z, -hw))], -2)
        if self.deformable and centre_feat is not None:
            raw = self.offset(centre_feat).view(rel.shape)
            rel = rel + self.d_max * torch.tanh(raw / self.d_max)
        return rel, box[..., None, :2] + _rot(box[..., 4, None], rel)

    def forward(self, feat, box, group, grid, dt=None):
        """feat (B, C, X, Y) BEV features; box (B, N, 7), group (B, N) -> list (n_iter) of dicts: box_in, delta
        (B, N, 7), quality (B, N) logit, box (B, N, 7) refined, points (B, N, 4, 2) m."""
        B, N = group.shape
        circle = self.circle_of_group[group]
        keep_c = torch.tensor([1, 1, 1, 0, 0, 1, 1], dtype=feat.dtype, device=feat.device)
        onehot = F.one_hot(group, len(self.groups)).to(feat.dtype)
        extra = []
        if self.use_dt:
            dt = torch.as_tensor(0.0 if dt is None else dt, dtype=feat.dtype, device=feat.device).reshape(-1).expand(B)
            extra.append(torch.stack([dt, dt.abs(), torch.sign(dt)], -1)[:, None].expand(B, N, 3))
        steps = []
        for _ in range(self.n_iter):
            cf = sample_bev(feat, box[..., None, :2], grid)[..., 0, :]
            rel, pts = self.points(box, group, cf)
            sf = sample_bev(feat, pts, grid)
            x = [cf, sf.flatten(-2), rel.flatten(-2), _rot(-box[..., 4], box[..., 5:7]) / 10, onehot] + extra
            if self.use_abs_xy:
                x.append(box[..., :2] / 50)
            out = self.mlp(torch.cat(x, -1))
            delta = out[..., :BOX_DIM] * torch.where(circle[..., None], keep_c, torch.ones_like(keep_c))
            nb = apply_residual(box, delta, circle)
            steps.append({"box_in": box, "delta": delta, "quality": out[..., BOX_DIM], "box": nb, "points": pts})
            box = nb.detach()
        return steps


@torch.no_grad()
def bev_iou(box, gt, group, mask, groups=GROUPS):
    """BEV IoU (shapely polygons; circles as 32-gons per quarter) of box and gt (B, N, 7) where mask > 0, else 0."""
    from shapely.geometry import Point, Polygon

    def shp(r, circle):
        cx, cy, a, b, yaw = r[:5]
        if circle:
            return Point(cx, cy).buffer(math.exp(a), 32)
        c, s = math.cos(yaw), math.sin(yaw)
        hl, hw = math.exp(a) / 2, math.exp(b) / 2
        return Polygon([(cx + c * u - s * w, cy + s * u + c * w) for u, w in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))])
    bl, gl, gr, ml = box.detach().cpu().tolist(), gt.detach().cpu().tolist(), group.cpu().tolist(), mask.cpu().tolist()
    out = torch.zeros(len(bl), len(bl[0]))
    for i in range(len(bl)):
        for j in range(len(bl[i])):
            if ml[i][j] > 0:
                circle = groups[gr[i][j]][1] == "circle"
                p, q = shp(bl[i][j], circle), shp(gl[i][j], circle)
                u = p.union(q).area
                out[i, j] = p.intersection(q).area / u if u > 0 else 0.0
    return out


def refine_candidates(dets, gts, k=50, n_jitter=1, jitter=(0.3, 0.1, 0.1, 0.5), match_dist=2.0, rng=None,
                      groups=GROUPS):
    """S5c training candidates per scene: the top-k decoded boxes (dicts, so detached) plus n_jitter copies of every GT
    box with Gaussian jitter (sigma: position m, log size, yaw rad, velocity m/s); every candidate is matched to the
    nearest GT of its group within match_dist m. Unobserved GT boxes (box.get("observed", True) False: no lidar /
    radar point, plan 8.4 masks) are ignored, not background: no jittered copies, and a candidate whose nearest GT is
    unobserved gets valid = matched = 0 (no regression, no quality term). Returns box, group, valid, gt (B, N, 7; the
    candidate itself where unmatched), matched (B, N) float."""
    import random
    rng = rng or random.Random(0)
    sp, ss, sy, sv = jitter
    cands, gtl, mat, ign = [], [], [], []
    for ds, gs in zip(dets, gts):
        cs = [dict(d) for d in sorted(ds, key=lambda d: -d.get("score", 0.0))[:k]]
        for g in gs:
            if not g.get("observed", True):
                continue
            for _ in range(n_jitter):
                j = dict(g, cx=g["cx"] + rng.gauss(0, sp), cy=g["cy"] + rng.gauss(0, sp), vx=g["vx"] + rng.gauss(0, sv),
                         vy=g["vy"] + rng.gauss(0, sv), score=1.0)
                if "radius" in g:
                    j["radius"] = g["radius"] * math.exp(rng.gauss(0, ss))
                else:
                    j.update(length=g["length"] * math.exp(rng.gauss(0, ss)), width=g["width"] * math.exp(rng.gauss(0, ss)),
                             yaw=g["yaw"] + rng.gauss(0, sy))
                cs.append(j)
        gm, mm, im = [], [], []
        for c in cs:
            best, bd = None, match_dist
            for g in gs:
                d = math.hypot(c["cx"] - g["cx"], c["cy"] - g["cy"])
                if g["group"] == c["group"] and d <= bd:
                    best, bd = g, d
            unobs = best is not None and not best.get("observed", True)
            gm.append(best if best is not None and not unobs else c)
            mm.append(0.0 if best is None or unobs else 1.0)
            im.append(unobs)
        cands.append(cs)
        gtl.append(gm)
        mat.append(mm)
        ign.append(im)
    box, group, valid = boxes_to_tensor(cands, groups)
    gt, _, _ = boxes_to_tensor(gtl, groups)
    matched = torch.zeros_like(valid)
    for i, (mm, im) in enumerate(zip(mat, ign)):
        matched[i, :len(mm)] = torch.tensor(mm)
        valid[i, :len(im)] *= 1.0 - torch.tensor(im, dtype=valid.dtype)
    return box, group, valid, gt, matched


def refine_loss(steps, gt, matched, valid, group, groups=GROUPS, w_vel=0.2, w_q=1.0):
    """S5c loss, summed over the iterations: L1 of the residuals against the targets GT - input box (matched
    candidates; circles without width and yaw; velocity weighted w_vel) + BCE of the quality logit against the BEV IoU
    of the input box with its GT (0 for unmatched candidates; all valid candidates). Ignored candidates (near an
    unobserved GT box, refine_candidates) have valid = matched = 0 and contribute nothing."""
    dev = gt.device
    circle = torch.tensor([k == "circle" for _, k in groups], device=dev)[group]
    wb = torch.tensor([1, 1, 1, 1, 1, w_vel, w_vel], dtype=gt.dtype, device=dev)
    wc = torch.tensor([1, 1, 1, 0, 0, w_vel, w_vel], dtype=gt.dtype, device=dev)
    w = torch.where(circle[..., None], wc, wb) * matched[..., None]
    reg, qual = 0.0, 0.0
    for st in steps:
        tgt = residual_targets(st["box_in"].detach(), gt, circle)
        reg = reg + (F.l1_loss(st["delta"], tgt, reduction="none") * w).sum() / matched.sum().clamp(min=1)
        iou = bev_iou(st["box_in"], gt, group, matched, groups).to(dev)
        bce = F.binary_cross_entropy_with_logits(st["quality"], iou, reduction="none")
        qual = qual + w_q * (bce * valid).sum() / valid.sum().clamp(min=1)
    return {"reg": reg, "quality": qual, "total": reg + qual}


@torch.no_grad()
def refine_boxes(refiner, feat, dets, grid, dt=None, rescore=False):
    """Run time: refine the decoded boxes (list (B) of lists of dicts) on the features feat (B, C, X, Y); rescore:
    score = sqrt(score * sigmoid(quality)) (CenterPoint's IoU-guided score)."""
    box, group, valid = boxes_to_tensor(dets, refiner.groups, feat.device)
    st = refiner(feat, box, group, grid, dt)[-1]
    res = tensor_to_boxes(st["box"], group, valid, refiner.groups, like=dets)
    if rescore:
        q = torch.sigmoid(st["quality"]).cpu()
        for i, boxes in enumerate(res):
            for j, bx in enumerate(boxes):
                bx["score"] = math.sqrt(bx["score"] * q[i, j].item())
    return res


# ----------------------------------------------------------------------------------------------- self-test
def _selftest():
    import sys
    sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
    from lift_pd import Grid
    torch.manual_seed(0)
    grid = Grid()
    boxes = [[{"group": "car", "cx": 20.3, "cy": 1.6, "length": 4.5, "width": 1.8, "yaw": 0.3, "vx": 8.0, "vy": 1.0},
              {"group": "truck", "cx": 41.1, "cy": -7.2, "length": 8.0, "width": 2.5, "yaw": -2.0, "vx": -5.0, "vy": 0.0},
              {"group": "pedestrian", "cx": 12.6, "cy": 9.9, "radius": 0.4, "vx": 0.0, "vy": 1.2}],
             [{"group": "car", "cx": 33.0, "cy": 0.2, "length": 4.5, "width": 1.8, "yaw": 3.0, "vx": 0.0, "vy": 0.0}]]
    t = build_targets(boxes, grid)
    assert t["hm"].max() == 1.0 and int(t["mask"].sum()) == 4
    # 1 perfect maps decode back to exactly the GT boxes
    dec = decode(targets_as_output(t, grid), grid, from_probabilities=True, threshold=0.99)
    for gt_list, d_list in zip(boxes, dec):
        assert len(d_list) == len(gt_list)
        for gb in gt_list:
            db = min(d_list, key=lambda d: math.hypot(d["cx"] - gb["cx"], d["cy"] - gb["cy"]))
            assert db["group"] == gb["group"] and math.hypot(db["cx"] - gb["cx"], db["cy"] - gb["cy"]) < 1e-4
            for k in ("length", "width", "radius", "vx", "vy"):
                if k in gb:
                    assert abs(db[k] - gb[k]) < 1e-4, k
            if "yaw" in gb:
                assert abs(math.remainder(db["yaw"] - gb["yaw"], 2 * math.pi)) < 1e-4
    # 2 the head runs, the loss is finite and the gradient reaches every branch
    head = ObjectHead(cin=16)
    out = head(torch.randn(2, 16, grid.nx, grid.ny))
    assert out["hm"].shape == (2, 3, 120, 104) and out["size"].shape == (2, 2, 120, 104)
    loss = head_loss(out, t)
    loss["total"].backward()
    for name in ("hm", "off", "size", "rot", "rad", "vel"):
        assert getattr(head, name)[2].weight.grad.abs().sum() > 0, name
    # 3 focal loss: a perfect prediction scores lower than a flat one
    flat = focal_loss(torch.full_like(t["hm"], -2.19), t["hm"])
    good = focal_loss(torch.logit(t["hm"].clamp(1e-4, 1 - 1e-4)), t["hm"])
    assert good < flat
    # 4 S5b: a box moving at 10 m/s, ego standing still: after 0.5 s it is 5 m further, heading and velocity kept;
    #   with ego yaw rate the heading turns by -omega dt; forward then backward returns to the start
    b0 = [{"group": "car", "cx": 20.0, "cy": 2.0, "length": 4.5, "width": 1.8, "yaw": 0.0, "vx": 10.0, "vy": 0.0}]
    b1 = propagate_boxes(b0, [(0.5, 0.0, 0.0)])[0]
    assert abs(b1["cx"] - 25.0) < 1e-9 and abs(b1["cy"] - 2.0) < 1e-9 and abs(b1["vx"] - 10.0) < 1e-9
    b2 = propagate_boxes(b0, [(0.5, 8.0, 0.4)])[0]
    assert abs(b2["yaw"] + 0.2) < 1e-9
    b3 = propagate_boxes([b2], [(-0.5, 8.0, 0.4)])[0]
    assert abs(b3["cx"] - 20.0) < 0.02 and abs(b3["cy"] - 2.0) < 0.02 and abs(b3["yaw"]) < 1e-9
    corr = BoxCorrection(cin=8)
    d, dy = corr(torch.randn(1, 3, 8), torch.randn(1, 3, 2), 0.5, 10.0, 0.1)
    assert d.abs().max() == 0 and dy.abs().max() == 0, "correction starts at zero"
    # 5 direction head (axis + front / back class): perfect maps decode to the GT yaw, also with the axis vector
    #   flipped (the class decides the half); the axis loss is sign-free; the class flips at DIR_OFFSET; gradients reach
    #   rot and dir
    assert direction_class(0.0) == 1.0 and direction_class(math.pi) == 0.0 and direction_class(-math.pi / 2) == 0.0
    assert direction_class(DIR_OFFSET + 1e-6) == 1.0 and direction_class(DIR_OFFSET - 1e-6) == 0.0
    po = targets_as_output(t, grid, direction=True)
    for flip in (1.0, -1.0):
        dec = decode(dict(po, rot=flip * po["rot"]), grid, from_probabilities=True, threshold=0.99)
        for gt_list, d_list in zip(boxes, dec):
            for gb in gt_list:
                if "yaw" in gb:
                    db = min(d_list, key=lambda d: math.hypot(d["cx"] - gb["cx"], d["cy"] - gb["cy"]))
                    assert abs(math.remainder(db["yaw"] - gb["yaw"], 2 * math.pi)) < 1e-4, (flip, db["yaw"], gb["yaw"])
    logit = lambda p: torch.logit(p.clamp(1e-4, 1 - 1e-4))                               # noqa: E731
    lp = head_loss(dict(po, hm=logit(po["hm"])), t)
    ln = head_loss(dict(po, hm=logit(po["hm"]), rot=-po["rot"]), t)
    assert lp["rot"] < 1e-6 and ln["rot"] < 1e-6 and lp["dir"] < 1e-3, (lp["rot"], ln["rot"], lp["dir"])
    wrong = head_loss(dict(po, hm=logit(po["hm"]), dir=-po["dir"]), t)
    assert wrong["dir"] > 1.0
    hd = ObjectHead(cin=16, direction=True)
    od = hd(torch.randn(2, 16, grid.nx, grid.ny))
    assert set(od) == {"hm", "off", "size", "rot", "rad", "vel", "dir"} and od["dir"].shape == (2, 1, 120, 104)
    head_loss(od, t)["total"].backward()
    assert hd.dir[2].weight.grad.abs().sum() > 0 and hd.rot[2].weight.grad.abs().sum() > 0
    assert "dir" not in out and "dir" not in loss                       # default head: unchanged outputs and terms
    print("object_head selftest ok: perfect targets decode to the GT boxes (centre, size, yaw, radius, velocity), "
          "shapes, finite loss with gradients in every branch, focal loss ordering, box propagation (S5b), "
          "correction zero start, direction head (classes, decode with either axis sign, sign-free axis loss, "
          "direction BCE, gradients)")
    _selftest_refiner(Grid)


def _selftest_refiner(Grid):
    """S5c: sampling convention, zero start, points on the faces rotating with the box, residual round trip, IoU,
    a toy case (every box shifted by +-0.6 m along and +-0.3 m across its axis) learned and undone, gradients."""
    import random
    torch.manual_seed(0)
    g = Grid(x_min=0.0, x_max=20.0, y_min=-10.0, y_max=10.0, res=0.25)
    xs, ys = g.xy_centres()
    # 1 bilinear sampling of linear maps returns the coordinates (x forward along dim 2, y left along dim 3)
    lin = torch.stack([xs.view(-1, 1).expand(g.nx, g.ny), ys.view(1, -1).expand(g.nx, g.ny)])[None]
    p = torch.tensor([[[[3.3, -2.1], [11.07, 4.6], [17.2, 0.01]]]])
    assert (sample_bev(lin, p, g) - p).abs().max() < 1e-4
    # 2 zero start: the boxes come back unchanged, the points are the face centres and rotate with the box
    gb = [[{"group": "car", "cx": 10.0, "cy": 1.0, "length": 4.5, "width": 1.8, "yaw": 0.3, "vx": 5.0, "vy": 1.0},
           {"group": "pedestrian", "cx": 6.0, "cy": -3.0, "radius": 0.4, "vx": 0.0, "vy": 1.0}]]
    box, group, valid = boxes_to_tensor(gb)
    ref = BoxRefiner(cin=2)
    st = ref(lin, box, group, g)[0]
    assert (st["box"] - box).abs().max() < 1e-6 and st["delta"].abs().max() == 0
    c, s = math.cos(0.3), math.sin(0.3)
    face = torch.tensor([[10 + 2.25 * c, 1 + 2.25 * s], [10 - 0.9 * s, 1 + 0.9 * c], [10 - 2.25 * c, 1 - 2.25 * s],
                         [10 + 0.9 * s, 1 - 0.9 * c]])
    assert (st["points"][0, 0] - face).abs().max() < 1e-5
    assert (st["points"][0, 1] - torch.tensor([[6.4, -3.0], [6.0, -2.6], [5.6, -3.0], [6.0, -3.4]])).abs().max() < 1e-5
    box2 = box.clone()
    box2[0, 0, 4] += 0.5
    pts2 = ref(lin, box2, group, g)[0]["points"][0, 0]
    ctr = torch.tensor([10.0, 1.0])
    assert (pts2 - (ctr + _rot(torch.tensor(0.5), face - ctr))).abs().max() < 1e-5
    # 3 residual targets and apply_residual are inverse; IoU of identical boxes 1, of a box shifted by l / 2 1 / 3
    gt = box + torch.tensor([0.4, -0.3, 0.1, -0.05, 0.2, 1.0, -0.5])
    circle = ref.circle_of_group[group]
    gt[0, 1, 3], gt[0, 1, 4] = gt[0, 1, 2], 0.0
    assert (apply_residual(box, residual_targets(box, gt, circle), circle) - gt).abs().max() < 1e-5
    assert abs(bev_iou(box, box, group, valid)[0, 0] - 1) < 1e-6
    sh = box.clone()
    sh[0, 0, 0] += 2.25 * c
    sh[0, 0, 1] += 2.25 * s
    assert abs(bev_iou(sh, box, group, valid)[0, 0] - 1 / 3) < 1e-4
    # 4 toy case: coverage maps of random cars; the candidates are the GT boxes shifted by (+-0.6, +-0.3) m in the
    #   box frame (random signs, so the refiner must read the samples); trained, it undoes the shift on fresh scenes
    rng = random.Random(0)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")

    def scene():
        boxes = []
        for _ in range(2):
            boxes.append({"group": "car", "cx": rng.uniform(4, 16), "cy": rng.uniform(-6, 6), "length": 4.5,
                          "width": 1.8, "yaw": rng.uniform(-math.pi, math.pi), "vx": 0.0, "vy": 0.0})
        cov = torch.zeros(g.nx, g.ny)
        for b in boxes:
            u = (gx - b["cx"]) * math.cos(b["yaw"]) + (gy - b["cy"]) * math.sin(b["yaw"])
            w = -(gx - b["cx"]) * math.sin(b["yaw"]) + (gy - b["cy"]) * math.cos(b["yaw"])
            cov = torch.maximum(cov, ((u.abs() <= b["length"] / 2) & (w.abs() <= b["width"] / 2)).float())
        shifted = []
        for b in boxes:
            d = _rot(torch.tensor(b["yaw"]), torch.tensor([rng.choice((-0.6, 0.6)), rng.choice((-0.3, 0.3))]))
            shifted.append(dict(b, cx=b["cx"] + d[0].item(), cy=b["cy"] + d[1].item()))
        return torch.stack([cov, torch.ones_like(cov)]), boxes, shifted

    def batch(n):
        sc = [scene() for _ in range(n)]
        feat = torch.stack([s[0] for s in sc])
        cb, cg, cv = boxes_to_tensor([s[2] for s in sc])
        gtb = boxes_to_tensor([s[1] for s in sc])[0]
        return feat, cb, cg, cv, gtb

    ref = BoxRefiner(cin=2, hidden=64)
    opt = torch.optim.Adam(ref.parameters(), 3e-3)
    for it in range(400):
        feat, cb, cg, cv, gtb = batch(16)
        loss = refine_loss(ref(feat, cb, cg, g), gtb, cv, cv, cg)
        opt.zero_grad()
        loss["total"].backward()
        if it == 2:                                   # zero start: from the third step on the gradient reaches all
            for name, prm in ref.named_parameters():
                assert prm.grad is not None and prm.grad.abs().sum() > 0, name
        opt.step()
    feat, cb, cg, cv, gtb = batch(64)
    with torch.no_grad():
        st = ref(feat, cb, cg, g)[0]
    e0 = (cb[..., :2] - gtb[..., :2]).norm(dim=-1).mean().item()
    e1 = (st["box"][..., :2] - gtb[..., :2]).norm(dim=-1).mean().item()
    iou0, iou1 = bev_iou(cb, gtb, cg, cv).mean().item(), bev_iou(st["box"], gtb, cg, cv).mean().item()
    moved = (ref.points(cb, cg, sample_bev(feat, cb[..., None, :2], g)[..., 0, :])[0]
             - ref.points(cb, cg)[0]).norm(dim=-1).mean().item()
    assert e1 < 0.4 * e0 and iou1 > iou0, (e0, e1, iou0, iou1)
    print(f"refiner selftest ok: sampling convention, zero start, face points rotate with the box, residual round "
          f"trip, IoU; toy shift undone on fresh scenes: centre error {e0:.3f} -> {e1:.3f} m, IoU {iou0:.3f} -> "
          f"{iou1:.3f} (mean point offset {moved:.3f} m); gradients reach the offsets and the MLP")


if __name__ == "__main__":
    _selftest()
