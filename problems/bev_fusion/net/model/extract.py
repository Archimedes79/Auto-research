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
Tests: tests/test_extract.py
"""
import math

import torch
import torch.nn.functional as F

from net.model.object_head import DIR_OFFSET


def _params(obj, ix, iy, grid, soft):
    """the box parameters of a set of cells from the head maps indexed on those cells: obj fields (B, C, *cells), ix / iy (B, *cells) the cell
    indices (same dtype as the maps). One code for the dense output (every cell: cells = (X, Y)) and for the selection (the K peak
    cells only: cells = (K,)). Returns cx, cy (metres, vehicle frame), length, width, radius, yaw, vx, vy and, soft, the heading vector."""
    out = {"cx": grid.x_min + (ix + obj["off"][:, 0] + 0.5) * grid.res, "cy": grid.y_min + (iy + obj["off"][:, 1] + 0.5) * grid.res}
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


def dense_boxes(obj, grid, groups, beta=10.0, soft=True, from_probabilities=False):
    """obj: the object head maps (hm logits (B, G, X, Y), off, size, rot, rad, vel; optional dir, mov) -> dict of dense maps:
    hm (heatmap in [0, 1]), w (peak weight), cx, cy, length, width, radius, yaw, vx, vy (each (B, X, Y); w and hm (B, G, X, Y))."""
    hm = obj["hm"] if from_probabilities else torch.sigmoid(obj["hm"])
    B, G, X, Y = hm.shape
    w = hm * torch.exp(-beta * (F.max_pool2d(hm, 3, 1, 1) - hm))
    ix = torch.arange(X, device=hm.device, dtype=hm.dtype).view(1, X, 1)
    iy = torch.arange(Y, device=hm.device, dtype=hm.dtype).view(1, 1, Y)
    return {"hm": hm, "w": w, **_params(obj, ix, iy, grid, soft)}


def _peaks(hm, k):
    """hard local maxima (3 x 3) of the heatmap (B, G, X, Y) -> top-k scores and the group / flat cell index of each, both (B, K)"""
    B, G, X, Y = hm.shape
    peaks = hm * (hm == F.max_pool2d(hm, 3, 1, 1)).to(hm.dtype)
    scores, idx = peaks.reshape(B, -1).topk(min(k, G * X * Y))
    return scores, torch.div(idx, X * Y, rounding_mode="floor"), idx % (X * Y)


_FIELDS = ("cx", "cy", "vx", "vy", "length", "width", "yaw", "radius")


def _to_dicts(scores, group, vals, fields, groups, threshold):
    """(B, K) scores / group index and (B, K, F) values -> list (B) of box dicts (object_head.decode's format), scores >= threshold; one
    transfer to the host for the whole batch, no per-box tensor access"""
    sc, gr, vl = scores.tolist(), group.tolist(), vals.tolist()
    col = {f: i for i, f in enumerate(fields)}
    res = []
    for b in range(len(sc)):
        boxes = []
        for s, g, v in zip(sc[b], gr[b], vl[b]):
            if s < threshold:
                break                                      # sorted: the rest is below as well
            bx = {"group": groups[g][0], "score": s, "cx": v[col["cx"]], "cy": v[col["cy"]], "vx": v[col["vx"]], "vy": v[col["vy"]]}
            if groups[g][1] == "box":
                bx.update(length=v[col["length"]], width=v[col["width"]], yaw=v[col["yaw"]])
            else:
                bx["radius"] = v[col["radius"]]
            boxes.append(bx)
        res.append(boxes)
    return res


def select_boxes(dense, groups, k=50, threshold=0.2):
    """the post-processing of a dense box output (planner side): hard local maximum of the heatmap, top-k, threshold -> list (B) of box
    dicts (object_head.decode's format); the K cells are gathered from the dense fields in one go"""
    scores, group, cell = _peaks(dense["hm"], k)
    fields = [f for f in _FIELDS if f in dense]
    vals = torch.stack([dense[f].reshape(dense[f].shape[0], -1).gather(1, cell) for f in fields], -1)
    return _to_dicts(scores, group, vals, fields, groups, threshold)


def boxes_from_maps(obj, grid, groups, k=50, threshold=0.2, from_probabilities=False):
    """THE box extractor: head maps -> list (B) of box dicts. Only the K peak cells are gathered from the raw maps (on the device of the
    maps) and parametrised, in float64, by the same `_params` as the dense output; object_head.decode, BEVFusionModel.boxes, the evaluation
    adapter and the prototype all come here. from_probabilities: hm already in [0, 1]."""
    hm = obj["hm"] if from_probabilities else torch.sigmoid(obj["hm"])
    scores, group, cell = _peaks(hm, k)
    Y = hm.shape[3]
    sub = {n: m.flatten(2).gather(2, cell[:, None, :].expand(-1, m.shape[1], -1)).double() for n, m in obj.items() if n != "hm"}
    p = _params(sub, torch.div(cell, Y, rounding_mode="floor").double(), (cell % Y).double(), grid, soft=False)
    fields = [f for f in _FIELDS if f in p]
    return _to_dicts(scores, group, torch.stack([p[f] for f in fields], -1), fields, groups, threshold)


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
