"""The targets and the losses of the object head (CenterPoint style: heat map, offset, size, heading, velocity, moving) and of the refiner's steps."""
import os as _os
import sys as _sys
_sys.path.insert(1, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # net/ (a flat run snapshot holds netpath.py next to this file)
import netpath  # noqa: E402,F401
import math

import numpy as np
import torch
import torch.nn.functional as F

import launch_saving as LS
from object_head import DIR_OFFSET, GROUPS, MOVING_SPEED, bev_iou, residual_targets


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
