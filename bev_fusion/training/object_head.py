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

import torch
import torch.nn as nn
import torch.nn.functional as F

GROUPS = (("car", "box"), ("truck", "box"), ("pedestrian", "circle"))     # (name, kind); synthetic-world classes


def _branch(cin, hidden, cout, bias=0.0):
    b = nn.Sequential(nn.Conv2d(cin, hidden, 3, padding=1), nn.ReLU(), nn.Conv2d(hidden, cout, 1))
    nn.init.constant_(b[2].bias, bias)
    return b


class ObjectHead(nn.Module):
    """BEV features (B, C, X, Y) -> dict of maps: hm (B, G, X, Y) logits, off (2), size (2: log l, log w),
    rot (2: sin, cos), rad (1: log r), vel (2)."""

    def __init__(self, cin, groups=GROUPS, hidden=64):
        super().__init__()
        self.groups = groups
        self.shared = nn.Sequential(nn.Conv2d(cin, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU())
        self.hm = _branch(hidden, hidden, len(groups), bias=-2.19)        # sigmoid(-2.19) = 0.1 (CenterNet init)
        self.off, self.size = _branch(hidden, hidden, 2), _branch(hidden, hidden, 2)
        self.rot, self.rad, self.vel = _branch(hidden, hidden, 2), _branch(hidden, hidden, 1), _branch(hidden, hidden, 2)

    def forward(self, feat):
        x = self.shared(feat)
        return {k: getattr(self, k)(x) for k in ("hm", "off", "size", "rot", "rad", "vel")}


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
    rad, vel, is_box)."""
    B, G, X, Y = len(boxes_batch), len(groups), grid.nx, grid.ny
    gidx = {name: i for i, (name, _) in enumerate(groups)}
    t = {"hm": torch.zeros(B, G, X, Y), "ind": torch.zeros(B, max_obj, dtype=torch.long),
         "mask": torch.zeros(B, max_obj), "group": torch.zeros(B, max_obj, dtype=torch.long),
         "off": torch.zeros(B, max_obj, 2), "size": torch.zeros(B, max_obj, 2), "rot": torch.zeros(B, max_obj, 2),
         "rad": torch.zeros(B, max_obj, 1), "vel": torch.zeros(B, max_obj, 2), "is_box": torch.zeros(B, max_obj)}
    gx = torch.arange(X, dtype=torch.float32).view(X, 1)
    gy = torch.arange(Y, dtype=torch.float32).view(1, Y)
    for b, boxes in enumerate(boxes_batch):
        n = 0
        for bx in boxes:
            fx = (bx["cx"] - grid.x_min) / grid.res - 0.5                 # continuous cell index of the centre
            fy = (bx["cy"] - grid.y_min) / grid.res - 0.5
            ix, iy = int(math.floor(fx + 0.5)), int(math.floor(fy + 0.5))
            if not (0 <= ix < X and 0 <= iy < Y) or n >= max_obj:
                continue
            g = gidx[bx["group"]]
            kind = groups[g][1]
            if kind == "box":
                r = gaussian_radius(bx["length"] / grid.res, bx["width"] / grid.res)
            else:
                d = 2 * bx["radius"] / grid.res
                r = gaussian_radius(d, d)
            r = max(min_radius, int(r))
            sigma = (2 * r + 1) / 6
            peak = torch.exp(-((gx - ix) ** 2 + (gy - iy) ** 2) / (2 * sigma * sigma))
            peak = peak * (((gx - ix).abs() <= r) & ((gy - iy).abs() <= r))
            t["hm"][b, g] = torch.maximum(t["hm"][b, g], peak)
            t["ind"][b, n], t["mask"][b, n], t["group"][b, n] = ix * Y + iy, 1.0, g
            t["off"][b, n] = torch.tensor([fx - ix, fy - iy])
            t["vel"][b, n] = torch.tensor([bx["vx"], bx["vy"]])
            if kind == "box":
                t["size"][b, n] = torch.tensor([math.log(bx["length"]), math.log(bx["width"])])
                t["rot"][b, n] = torch.tensor([math.sin(bx["yaw"]), math.cos(bx["yaw"])])
                t["is_box"][b, n] = 1.0
            else:
                t["rad"][b, n] = math.log(bx["radius"])
            n += 1
    return t


# ------------------------------------------------------------------------------------------------ losses
def focal_loss(logits, target):
    """CenterNet focal loss (alpha 2, beta 4), normalised by the number of peaks."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg = 1 - pos
    pos_l = -torch.log(p) * (1 - p) ** 2 * pos
    neg_l = -torch.log(1 - p) * p ** 2 * (1 - target) ** 4 * neg
    return (pos_l.sum() + neg_l.sum()) / pos.sum().clamp(min=1)


def _gather(m, ind):
    """m (B, C, X, Y), ind (B, M) flat cell index -> (B, M, C)."""
    B, C = m.shape[:2]
    return m.reshape(B, C, -1).gather(2, ind.unsqueeze(1).expand(B, C, ind.shape[1])).permute(0, 2, 1)


def head_loss(out, t, w_vel=0.2):
    """dict of loss terms and their sum 'total'."""
    dev = out["hm"].device
    t = {k: v.to(dev) for k, v in t.items()}
    m = t["mask"].unsqueeze(-1)
    mb = (t["mask"] * t["is_box"]).unsqueeze(-1)
    mc = (t["mask"] * (1 - t["is_box"])).unsqueeze(-1)

    def l1(name, mask):
        return (F.l1_loss(_gather(out[name], t["ind"]), t[name], reduction="none") * mask).sum() / mask.sum().clamp(min=1)
    rot = _gather(out["rot"], t["ind"])
    rot = rot / rot.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    losses = {"hm": focal_loss(out["hm"], t["hm"]), "off": l1("off", m), "size": l1("size", mb),
              "rot": (F.l1_loss(rot, t["rot"], reduction="none") * mb).sum() / mb.sum().clamp(min=1),
              "rad": l1("rad", mc), "vel": w_vel * l1("vel", m)}
    losses["total"] = sum(losses.values())
    return losses


# ------------------------------------------------------------------------------------------------ decoding
@torch.no_grad()
def decode(out, grid, groups=GROUPS, k=50, threshold=0.2, from_probabilities=False):
    """out maps -> list (B) of box dicts with 'score'. from_probabilities: hm already in [0, 1] (self-test)."""
    hm = out["hm"] if from_probabilities else torch.sigmoid(out["hm"])
    keep = (hm == F.max_pool2d(hm, 3, 1, 1)).float()
    hm = hm * keep
    B, G, X, Y = hm.shape
    scores, idx = hm.reshape(B, -1).topk(min(k, G * X * Y))
    res = []
    for b in range(B):
        boxes = []
        for s, i in zip(scores[b].tolist(), idx[b].tolist()):
            if s < threshold:
                continue
            g, cell = divmod(i, X * Y)
            ix, iy = divmod(cell, Y)
            ox, oy = out["off"][b, :, ix, iy].tolist()
            bx = {"group": groups[g][0], "score": s,
                  "cx": grid.x_min + (ix + ox + 0.5) * grid.res, "cy": grid.y_min + (iy + oy + 0.5) * grid.res,
                  "vx": out["vel"][b, 0, ix, iy].item(), "vy": out["vel"][b, 1, ix, iy].item()}
            if groups[g][1] == "box":
                ll, lw = out["size"][b, :, ix, iy].tolist()
                sn, cs = out["rot"][b, :, ix, iy].tolist()
                bx.update(length=math.exp(ll), width=math.exp(lw), yaw=math.atan2(sn, cs))
            else:
                bx["radius"] = math.exp(out["rad"][b, 0, ix, iy].item())
            boxes.append(bx)
        res.append(boxes)
    return res


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


def targets_as_output(t, grid):
    """Perfect head output from targets (self-test): hm as probabilities, regressions written at the centre cells."""
    B, G, X, Y = t["hm"].shape
    out = {"hm": t["hm"].clone()}
    for name, c in (("off", 2), ("size", 2), ("rot", 2), ("rad", 1), ("vel", 2)):
        m = torch.zeros(B, c, X * Y)
        for b in range(B):
            for n in range(t["ind"].shape[1]):
                if t["mask"][b, n] > 0:
                    m[b, :, t["ind"][b, n]] = t[name][b, n]
        out[name] = m.view(B, c, X, Y)
    return out


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
    print("object_head selftest ok: perfect targets decode to the GT boxes (centre, size, yaw, radius, velocity), "
          "shapes, finite loss with gradients in every branch, focal loss ordering, box propagation (S5b), "
          "correction zero start")


if __name__ == "__main__":
    _selftest()
