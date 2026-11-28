"""Tests of net/model/object_head.py (its former self-test).

    python tests/test_object_head.py      (from the problem folder, with its module path)
"""
import math

import torch

from net.model.object_head import BoxRefiner, DIR_OFFSET, ObjectHead, _rot, apply_residual, bev_iou, boxes_to_tensor, decode, direction_class, propagate_boxes, residual_targets, sample_bev, targets_as_output


def _selftest():
    from net.loss.object_loss import build_targets, focal_loss, head_loss
    from net.model.lift_pd import Grid
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
          "direction head (classes, decode with either axis sign, sign-free axis loss, "
          "direction BCE, gradients)")
    _selftest_refiner(Grid)


def _selftest_refiner(Grid):
    """S5c: sampling convention, zero start, points on the faces rotating with the box, residual round trip, IoU,
    a toy case (every box shifted by +-0.6 m along and +-0.3 m across its axis) learned and undone, gradients."""
    from net.loss.object_loss import refine_loss
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
