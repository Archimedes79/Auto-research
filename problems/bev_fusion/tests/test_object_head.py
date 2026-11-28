"""Tests of net/model/object_head.py (its former self-test).

    python -P -m bev_fusion.tests.test_object_head      (from the problem folder, with its module path)
"""
import math

import torch

from bev_fusion.net.model.object_head import DIR_OFFSET, ObjectHead, decode, direction_class


def targets_as_output(t, grid, direction=False):
    """Perfect head output from targets: hm as probabilities, regressions written at the centre cells;
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


def _selftest():
    from bev_fusion.net.loss.object_loss import build_targets, focal_loss, head_loss
    from bev_fusion.net.model.lift_pd import Grid
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
    # 4 direction head (axis + front / back class): perfect maps decode to the GT yaw, also with the axis vector
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
          "shapes, finite loss with gradients in every branch, focal loss ordering, "
          "direction head (classes, decode with either axis sign, sign-free axis loss, "
          "direction BCE, gradients)")


if __name__ == "__main__":
    _selftest()
