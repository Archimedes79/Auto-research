"""One-off patch (open files only: losses.py, object_head.py): the host side of the loss without changing one bit.
 1. losses.py: `_stack_many` stacks every GT layer of all samples on the HOST, joins the layers along the channel axis and moves them with ONE copy
    (the step did ~1300 copies, one per layer per sample: ~0.4 s of 4.1 s, records/speed_options_2026-11-03.md); build_layer_targets uses it.
 2. object_head.py: `build_targets` fills numpy arrays per frame (no torch.tensor per field per box) and exponentiates the Gaussian windows of a frame in
    ONE torch.exp call (numpy's exp differs from torch's in the last bit for ~40 % of the values: torch stays, so the heatmaps are bit-identical).
Exact-marker replacements, each must match once."""
import os

T = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "training"))


def rd(n):
    return open(os.path.join(T, n), encoding="utf-8").read()


def wr(n, s):
    open(os.path.join(T, n), "w", encoding="utf-8", newline="\n").write(s)


# ------------------------------------------------------------------ losses.py
s = rd("losses.py")
old = '''def build_layer_targets(gts, cfg=None, mass=None, dt=0.0, device="cpu"):'''
assert s.count(old) == 1
NEW = '''def _stack_many(gts, keys, device):
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


'''
s = s.replace(old, NEW + old, 1)
old = '''    T = {}
    rows = layer_table(cfg.centrelines, True, cfg.centreline_1m)          # the objects are always read: the aux heads' targets
    for k in source_keys(rows) + ["velocity"]:
        T[k] = _stack(gts, k, device)
    if cfg.radar_stationary and all("radar_stationary" in g for g in gts):       # hook: another session builds it
        T["radar_stationary"] = _stack(gts, "radar_stationary", device)
    if cfg.hygiene_2026_11_02 and all("flow_weight" in g for g in gts):          # autogt: 1 where the velocity is known, 0 on the cells of a box of unknown velocity (velocity 0 there)
        T["flow_weight"] = _stack(gts, "flow_weight", device)
    T["obj_cells"] = (_stack(gts, "instance", device) >= 0).float()
    cam, occ = _stack(gts, "camera", device), _stack(gts, "occluded", device)
    radar, undef = _stack(gts, "radar", device), _stack(gts, "undefined", device)
'''
assert s.count(old) == 1
new = '''    T = {}
    rows = layer_table(cfg.centrelines, True, cfg.centreline_1m)          # the objects are always read: the aux heads' targets
    has_stat = cfg.radar_stationary and all("radar_stationary" in g for g in gts)       # hook: another session builds it
    has_flow = cfg.hygiene_2026_11_02 and all("flow_weight" in g for g in gts)          # autogt: 1 where the velocity is known, 0 on the cells of a box of unknown velocity (velocity 0 there)
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
'''
s = s.replace(old, new, 1)
old = '''    keep = (1 - _stack(gts, "ignore", device).clamp(0, 1)) if all("ignore" in g for g in gts) else torch.ones_like(cam)'''
assert s.count(old) == 1
s = s.replace(old, '''    keep = (1 - S["ignore"].clamp(0, 1)) if has_ign else torch.ones_like(cam)''', 1)
wr("losses.py", s)

# ------------------------------------------------------------------ object_head.py
o = rd("object_head.py")
a = o.index("def build_targets(boxes_batch, grid, groups=GROUPS, max_obj=64, min_radius=2):")
b = o.index("# ------------------------------------------------------------------------------------------------ losses")
assert o.count("# ------------------------------------------------------------------------------------------------ losses") == 1
NEWBT = '''def build_targets(boxes_batch, grid, groups=GROUPS, max_obj=64, min_radius=2):
    """boxes_batch: list (B) of lists of box dicts -> dict of target tensors (hm, ind, mask, group, off, size, rot,
    rad, vel, is_box, dir). The per-box fields are made for a whole frame at once in numpy and handed over with one torch.from_numpy per field (the
    version before made a torch.tensor per field per box: 2.6 x slower); the Gaussian windows of a frame are exponentiated in ONE torch.exp call, and
    torch.exp stays: numpy's exp differs from it in the last bit for ~40 % of the values, so the heatmaps are bit-identical to the old version
    (tests/test_host_targets.py checks all tensors with torch.equal against a copy of the old function)."""
    B, G, X, Y = len(boxes_batch), len(groups), grid.nx, grid.ny
    gidx = {name: i for i, (name, _) in enumerate(groups)}
    box_kind = np.array([k == "box" for _, k in groups])
    hm = np.zeros((B, G, X, Y), np.float32)
    ind, grp = np.zeros((B, max_obj), np.int64), np.zeros((B, max_obj), np.int64)
    mask, is_box = np.zeros((B, max_obj), np.float32), np.zeros((B, max_obj), np.float32)
    off, size, rot, vel = (np.zeros((B, max_obj, 2), np.float32) for _ in range(4))
    rad, dcl = np.zeros((B, max_obj, 1), np.float32), np.zeros((B, max_obj, 1), np.float32)
    for b, boxes in enumerate(boxes_batch):
        if not boxes:
            continue
        g = np.array([gidx[bx["group"]] for bx in boxes])
        fx = (np.array([bx["cx"] for bx in boxes]) - grid.x_min) / grid.res - 0.5     # continuous cell index of the centre
        fy = (np.array([bx["cy"] for bx in boxes]) - grid.y_min) / grid.res - 0.5
        ix, iy = np.floor(fx + 0.5).astype(np.int64), np.floor(fy + 0.5).astype(np.int64)
        sel = np.nonzero((ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y))[0][:max_obj]
        n = len(sel)
        if not n:
            continue
        g, fx, fy, ix, iy = g[sel], fx[sel], fy[sel], ix[sel], iy[sel]
        bb = [boxes[i] for i in sel]
        isb = box_kind[g]
        length = np.array([bx["length"] if k else 1.0 for bx, k in zip(bb, isb)])
        width = np.array([bx["width"] if k else 1.0 for bx, k in zip(bb, isb)])
        radius = np.array([1.0 if k else bx["radius"] for bx, k in zip(bb, isb)])
        yaw = np.array([bx["yaw"] if k else 0.0 for bx, k in zip(bb, isb)])
        r = np.array([gaussian_radius(ln / grid.res, w / grid.res) if k else gaussian_radius(2 * rd / grid.res, 2 * rd / grid.res)
                      for ln, w, rd, k in zip(length, width, radius, isb)])
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
        ind[b, :n], mask[b, :n], grp[b, :n] = ix * Y + iy, 1.0, g
        off[b, :n] = np.stack([fx - ix, fy - iy], 1)
        vel[b, :n] = np.stack([[bx["vx"] for bx in bb], [bx["vy"] for bx in bb]], 1)
        size[b, :n] = np.where(isb[:, None], np.log(np.stack([length, width], 1)), 0.0)
        rot[b, :n] = np.where(isb[:, None], np.stack([np.sin(yaw), np.cos(yaw)], 1), 0.0)
        dcl[b, :n, 0] = np.where(isb, ((yaw - DIR_OFFSET) % (2 * math.pi) < math.pi), 0.0)
        rad[b, :n, 0] = np.where(isb, 0.0, np.log(radius))
        is_box[b, :n] = isb
    return {k: torch.from_numpy(v) for k, v in dict(hm=hm, ind=ind, mask=mask, group=grp, off=off, size=size, rot=rot, rad=rad, vel=vel,
                                                     is_box=is_box, dir=dcl).items()}


'''
o = o[:a] + NEWBT + o[b:]
if "\nimport numpy as np\n" not in o:
    o = o.replace("import math\n", "import math\n\nimport numpy as np\n", 1)
wr("object_head.py", o)
print("patched losses.py, object_head.py")
