"""Tests of net/model/extract.py (its former self-test).

    python tests/test_extract.py      (from the problem folder, with its module path)
"""
import math

import torch
import torch.nn.functional as F

from net.model.object_head import DIR_OFFSET
from net.model.extract import boxes_from_maps, dense_boxes, select_boxes, soft_line_map


def _decode_reference(out, grid, groups, k=50, threshold=0.2):
    """the plain per-box loop of the old object_head.decode: the independent oracle of the self-test (python floats, one box at a time)"""
    hm = torch.sigmoid(out["hm"])
    hm = hm * (hm == F.max_pool2d(hm, 3, 1, 1)).float()
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
            bx = {"group": groups[g][0], "score": s, "cx": grid.x_min + (ix + ox + 0.5) * grid.res, "cy": grid.y_min + (iy + oy + 0.5) * grid.res,
                  "vx": out["vel"][b, 0, ix, iy].item(), "vy": out["vel"][b, 1, ix, iy].item()}
            if "mov" in out and out["mov"][b, 0, ix, iy].item() < 0:
                bx["vx"] = bx["vy"] = 0.0
            if groups[g][1] == "box":
                ll, lw = out["size"][b, :, ix, iy].tolist()
                sn, cs = out["rot"][b, :, ix, iy].tolist()
                yaw = math.atan2(sn, cs)
                if "dir" in out:
                    yaw = (yaw - DIR_OFFSET) % math.pi + DIR_OFFSET
                    if out["dir"][b, 0, ix, iy].item() < 0:
                        yaw = math.remainder(yaw + math.pi, 2 * math.pi)
                bx.update(length=math.exp(ll), width=math.exp(lw), yaw=yaw)
            else:
                bx["radius"] = math.exp(out["rad"][b, 0, ix, iy].item())
            boxes.append(bx)
        res.append(boxes)
    return res


def _selftest():
    import numpy as np
    from evaluation.metrics import line_cells

    class Grid:
        x_min, y_min, res = -10.0, -12.0, 0.5

    groups = (("vehicle", "box"), ("two_wheeler", "box"), ("pedestrian", "circle"))
    torch.manual_seed(0)
    B, G, X, Y = 2, 3, 40, 48
    obj = {"hm": torch.randn(B, G, X, Y) * 2 - 1, "off": torch.rand(B, 2, X, Y), "size": torch.randn(B, 2, X, Y) * 0.3,
           "rot": torch.randn(B, 2, X, Y), "rad": torch.randn(B, 1, X, Y) * 0.2, "vel": torch.randn(B, 2, X, Y),
           "dir": torch.randn(B, 1, X, Y), "mov": torch.randn(B, 1, X, Y)}
    obj = {k: v.double() for k, v in obj.items()}
    # 1. the vectorised selection (from the dense output AND straight from the maps, = object_head.decode) reproduces the plain loop box by box
    ref = _decode_reference(obj, Grid, groups, k=60, threshold=0.2)
    got = select_boxes(dense_boxes(obj, Grid, groups, soft=False), groups, k=60, threshold=0.2)
    direct = boxes_from_maps(obj, Grid, groups, k=60, threshold=0.2)
    assert [len(a) for a in direct] == [len(b) for b in got] == [len(r) for r in ref], "the two entry points select the same boxes"
    for db, gb in zip(direct, got):
        for d_, g_ in zip(db, gb):
            assert d_.keys() == g_.keys() and all(abs(d_[kk] - g_[kk]) < 1e-9 for kk in d_ if kk != "group") and d_["group"] == g_["group"]
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
    r32 = _decode_reference(o32, Grid, groups, k=60, threshold=0.2)
    g32 = boxes_from_maps(o32, Grid, groups, k=60, threshold=0.2)
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
    print("extract selftest passed: the vectorised selection = the plain per-box loop on", sum(len(r) for r in ref),
          "boxes, dense boxes differentiable, soft line map >= 0.5 = metrics.line_cells")


if __name__ == "__main__":
    _selftest()
