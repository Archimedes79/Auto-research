"""Differentiable extraction (layer 4 of training/architecture.md): the head maps -> dense outputs a planner can
read and train through, and the one small post-processing step that picks discrete boxes and line cells for the metrics.

    dense = dense_boxes(obj, grid, groups)          every cell: peak weight w (B, G, X, Y) and the box it would decode to
    boxes = select_boxes(dense, grid, groups)       local maximum, top-k, threshold -> the box dicts of object_head.decode
    p = soft_line_map(dist, res)                    (..., X, Y) in [0, 1]; p >= 0.5 gives exactly evaluation/metrics.line_cells

Boxes: the peak weight is w = hm * exp(-beta * (maxpool3(hm) - hm)): the heatmap at a local maximum, smoothly lower elsewhere.
The parameters per cell are those the head regresses there: centre = cell + offset (metres, vehicle frame), length / width
(exp of the log size), radius, yaw from (sin, cos) - with the direction class the axis is mapped into [DIR_OFFSET, DIR_OFFSET + pi)
and turned by pi where the class says "back" - and the velocity. soft=True (the dense output) adds the heading vector (hx, hy) = the
axis times tanh(dir logit) (its length is the front / back confidence; the angle `yaw` itself cannot carry a gradient to the class)
and scales the velocity by sigmoid(moving logit), both differentiable; soft=False (selection): the hard choices of
object_head.decode, so select_boxes(dense_boxes(..., soft=False)) returns its boxes (call it on double maps for exact equality).
Lines: from the predicted distance field (metres): near = sigmoid(k (res/2 - d)), a local minimum along x or y = sigmoid(k
(min of the two neighbours - d)); p = min(near, max(min_x, min_y)), so p >= 0.5 exactly when line_cells marks the cell.
Run the self-test: python training/extract.py
"""
import math

import torch
import torch.nn.functional as F

try:
    from .object_head import DIR_OFFSET
except ImportError:
    from object_head import DIR_OFFSET


def _centres(grid, X, Y, off):
    """cell index + offset -> metres (vehicle frame), as object_head.decode"""
    ix = torch.arange(X, device=off.device, dtype=off.dtype).view(1, X, 1)
    iy = torch.arange(Y, device=off.device, dtype=off.dtype).view(1, 1, Y)
    return grid.x_min + (ix + off[:, 0] + 0.5) * grid.res, grid.y_min + (iy + off[:, 1] + 0.5) * grid.res


def dense_boxes(obj, grid, groups, beta=10.0, soft=True, from_probabilities=False):
    """obj: the object head maps (hm logits (B, G, X, Y), off, size, rot, rad, vel; optional dir, mov) -> dict of dense maps:
    hm (heatmap in [0, 1]), w (peak weight), cx, cy, length, width, radius, yaw, vx, vy (each (B, X, Y); w and hm (B, G, X, Y))."""
    hm = obj["hm"] if from_probabilities else torch.sigmoid(obj["hm"])
    B, G, X, Y = hm.shape
    w = hm * torch.exp(-beta * (F.max_pool2d(hm, 3, 1, 1) - hm))
    cx, cy = _centres(grid, X, Y, obj["off"])
    out = {"hm": hm, "w": w, "cx": cx, "cy": cy}
    if "size" in obj:
        out["length"], out["width"] = torch.exp(obj["size"][:, 0]), torch.exp(obj["size"][:, 1])
    if "rad" in obj:
        out["radius"] = torch.exp(obj["rad"][:, 0])
    if "rot" in obj:
        yaw = torch.atan2(obj["rot"][:, 0], obj["rot"][:, 1])
        if "dir" in obj:                                   # the box axis in [off, off + pi), then the half from the class
            axis = torch.remainder(yaw - DIR_OFFSET, math.pi) + DIR_OFFSET
            back = obj["dir"][:, 0] < 0
            if soft:                                       # heading vector: the axis times tanh(dir) - its length is the front / back
                t = torch.tanh(obj["dir"][:, 0])           # confidence, so the gradient reaches the dir logit (an angle cannot carry it)
                out["hx"], out["hy"] = t * torch.cos(axis), t * torch.sin(axis)
            yaw = torch.where(back, torch.remainder(axis + 2 * math.pi, 2 * math.pi) - math.pi, axis)
        elif soft:
            out["hx"], out["hy"] = torch.cos(yaw), torch.sin(yaw)
        out["yaw"] = yaw
    v = obj["vel"]
    if "mov" in obj:                                       # static (moving logit < 0): velocity 0 (the AVE-optimal answer)
        gate = torch.sigmoid(obj["mov"]) if soft else (obj["mov"] >= 0).to(v.dtype)
        v = v * gate
    out["vx"], out["vy"] = v[:, 0], v[:, 1]
    return out


def select_boxes(dense, groups, k=50, threshold=0.2):
    """the post-processing: hard local maximum of the heatmap, top-k, threshold -> list (B) of box dicts (object_head.decode's format)"""
    hm = dense["hm"]
    B, G, X, Y = hm.shape
    peaks = hm * (hm == F.max_pool2d(hm, 3, 1, 1)).to(hm.dtype)
    scores, idx = peaks.reshape(B, -1).topk(min(k, G * X * Y))
    res = []
    for b in range(B):
        boxes = []
        for s, i in zip(scores[b].tolist(), idx[b].tolist()):
            if s < threshold:
                continue
            g, cell = divmod(i, X * Y)
            ix, iy = divmod(cell, Y)
            bx = {"group": groups[g][0], "score": s, "cx": dense["cx"][b, ix, iy].item(), "cy": dense["cy"][b, ix, iy].item(),
                  "vx": dense["vx"][b, ix, iy].item(), "vy": dense["vy"][b, ix, iy].item()}
            if groups[g][1] == "box":
                bx.update(length=dense["length"][b, ix, iy].item(), width=dense["width"][b, ix, iy].item(), yaw=dense["yaw"][b, ix, iy].item())
            else:
                bx["radius"] = dense["radius"][b, ix, iy].item()
            boxes.append(bx)
        res.append(boxes)
    return res


def soft_line_map(dist, res, k=None):
    """distance field (..., X, Y) in metres -> line probability (..., X, Y); k: the sharpness (default 50 / res)"""
    k = 50.0 / res if k is None else k
    d = dist
    pad = F.pad(d.reshape(-1, 1, *d.shape[-2:]), (1, 1, 1, 1), value=float("inf")).reshape(*d.shape[:-2], d.shape[-2] + 2, d.shape[-1] + 2)
    nx = torch.minimum(pad[..., :-2, 1:-1], pad[..., 2:, 1:-1])
    ny = torch.minimum(pad[..., 1:-1, :-2], pad[..., 1:-1, 2:])
    near = torch.sigmoid(k * (res / 2 + 1e-9 - d))
    loc = torch.maximum(torch.sigmoid(k * (nx - d)), torch.sigmoid(k * (ny - d)))
    return torch.minimum(near, loc)


# ----------------------------------------------------------------------------------------------- self-test
def _selftest():
    import numpy as np
    try:
        from .object_head import decode
    except ImportError:
        from object_head import decode
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evaluation"))
    from metrics import line_cells

    class Grid:
        x_min, y_min, res = -10.0, -12.0, 0.5

    groups = (("vehicle", "box"), ("two_wheeler", "box"), ("pedestrian", "circle"))
    torch.manual_seed(0)
    B, G, X, Y = 2, 3, 40, 48
    obj = {"hm": torch.randn(B, G, X, Y) * 2 - 1, "off": torch.rand(B, 2, X, Y), "size": torch.randn(B, 2, X, Y) * 0.3,
           "rot": torch.randn(B, 2, X, Y), "rad": torch.randn(B, 1, X, Y) * 0.2, "vel": torch.randn(B, 2, X, Y),
           "dir": torch.randn(B, 1, X, Y), "mov": torch.randn(B, 1, X, Y)}
    obj = {k: v.double() for k, v in obj.items()}
    # 1. the hard selection reproduces object_head.decode box by box
    ref = decode(obj, Grid, groups, k=60, threshold=0.2)
    got = select_boxes(dense_boxes(obj, Grid, groups, soft=False), groups, k=60, threshold=0.2)
    assert [len(r) for r in ref] == [len(g) for g in got] and sum(len(r) for r in ref) > 20, ([len(r) for r in ref], [len(g) for g in got])
    for rb, gb in zip(ref, got):
        for r, g in zip(rb, gb):
            assert r.keys() == g.keys() and r["group"] == g["group"], (r, g)
            for key in r:
                if key == "group":
                    continue
                dv = r[key] - g[key]
                if key == "yaw":
                    dv = math.remainder(dv, 2 * math.pi)
                assert abs(dv) < 1e-9, (key, r[key], g[key])
    # 2. the dense output is differentiable (each head map gets a real gradient, the dir logit through the heading vector,
    #    also at dir = 0) and equals the heatmap at its peaks
    obj_g = {kk: v.clone().requires_grad_(True) for kk, v in obj.items()}
    with torch.no_grad():
        obj_g["dir"][0, 0, :4, :4] = 0.0                          # exactly undecided cells: no NaN
    dense = dense_boxes(obj_g, Grid, groups, soft=True)
    total = sum(dense[kk].sum() for kk in ("w", "cx", "cy", "length", "width", "radius", "hx", "hy", "vx", "vy"))
    total.backward()
    for kk, v in obj_g.items():
        assert v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().max() > 1e-3, (kk, v.grad.abs().max())
    # 1b. float32 maps (the model's dtype): the same boxes as decode on the same float32 maps, values within float32 rounding
    o32 = {kk: v.float() for kk, v in obj.items()}
    r32 = decode(o32, Grid, groups, k=60, threshold=0.2)
    g32 = select_boxes(dense_boxes({kk: v.double() for kk, v in o32.items()}, Grid, groups, soft=False), groups, k=60, threshold=0.2)
    assert [[b["group"] for b in r] for r in r32] == [[b["group"] for b in g] for g in g32]
    assert max(abs(math.remainder(r["yaw"] - g["yaw"], 2 * math.pi)) for rb, gb in zip(r32, g32) for r, g in zip(rb, gb) if "yaw" in r) < 1e-6
    hm = dense["hm"].detach()
    peak = hm == F.max_pool2d(hm, 3, 1, 1)
    assert torch.equal(dense["w"].detach()[peak], hm[peak]) and (dense["w"].detach() <= hm + 1e-12).all()
    # 3. the soft line map thresholded at 0.5 gives exactly line_cells (random fields with exact ties and a planted line)
    for seed in range(5):
        g = torch.Generator().manual_seed(seed)
        d = torch.rand(X, Y, generator=g, dtype=torch.float64) * 2.0
        d[10, :] = 0.0                                            # a straight line of zeros: ties between neighbours
        d[:, 7] = torch.round(d[:, 7] * 4) / 4                    # quantised column: more exact ties
        p = soft_line_map(d, Grid.res)
        assert np.array_equal(p.numpy() >= 0.5, line_cells(d.numpy(), Grid.res)), seed
        neg = d - 0.3                                             # a head can predict negative distances: the model and the
        p = soft_line_map(neg.clamp(min=0), Grid.res)             # evaluation (bev_adapter) clamp them at 0 first
        assert np.array_equal(p.numpy() >= 0.5, line_cells(neg.clamp(min=0).numpy(), Grid.res)), seed
    dd = d.clone().requires_grad_(True)
    soft_line_map(dd, Grid.res).sum().backward()
    assert torch.isfinite(dd.grad).all() and dd.grad.abs().sum() > 0
    print("extract selftest passed: hard selection = object_head.decode on", sum(len(r) for r in ref),
          "boxes, dense boxes differentiable, soft line map >= 0.5 = metrics.line_cells")


if __name__ == "__main__":
    _selftest()
