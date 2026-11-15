"""Fix of the critic's findings 1, 3, 4, 5 in object_head.build_targets: loss_w and vel_known (the hygiene flag's inputs) are handled again, the group is
looked up after the range filter and the cap (as before), degenerate values still raise, per-box fields are float64 (python float semantics)."""
import os

P = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "training", "object_head.py"))
o = open(P, encoding="utf-8").read()
a = o.index("def build_targets(boxes_batch, grid, groups=GROUPS, max_obj=64, min_radius=2):")
b = o.index("# ------------------------------------------------------------------------------------------------ losses")
NEW = '''def build_targets(boxes_batch, grid, groups=GROUPS, max_obj=64, min_radius=2):
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


'''
o = o[:a] + NEW + o[b:]
open(P, "w", encoding="utf-8", newline="\n").write(o)
print("patched build_targets")
