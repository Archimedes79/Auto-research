"""Tests of net/model/lane_branch.py (its former self-test).

    python -P -m bev_fusion.tests.test_lane_branch      (from the problem folder, with its module path)
"""
import os

import torch

from bev_fusion.net import netpath
import torch.nn.functional as F

from bev_fusion.net.model.lane_branch import DUALBEV_CKPT, LaneBranch, Res4Features
from bev_fusion.net.model.lift_pd import LiftPD


def _selftest():
    from bev_fusion.net.model.lift_pd import Grid, _front_camera
    torch.manual_seed(0)
    grid = Grid(z_levels=(-0.5, 0.0, 0.5, 1.0))
    depths = torch.arange(1.0, 60.0, 0.5)
    K, T, hw = _front_camera(torch.float32)
    H, W = hw

    # 1. shapes, zero start, gradients after one step (random frozen features: no checkpoint needed)
    br = LaneBranch(grid, depths, c_out=8, oversample=2, pretrained="")
    img = torch.randint(0, 256, (2, 3, H, W), dtype=torch.uint8)
    prob = torch.rand(2, len(depths), H // 16, W // 16).softmax(1)
    Kb, Tb = K.expand(2, 3, 3), T.expand(2, 4, 4)
    out = br(img, prob, Kb, Tb, hw)
    assert out.shape == (2, 8, grid.nx, grid.ny), out.shape
    assert out.abs().max().item() == 0.0, "zero start: the branch must add nothing at step 0"
    opt = torch.optim.SGD(br.parameters(), lr=0.1)
    target = torch.randn_like(out)
    for _ in range(2):
        opt.zero_grad()
        F.mse_loss(br(img, prob, Kb, Tb, hw), target).backward()
        opt.step()
    g = [p.grad for p in br.reduce.parameters()]
    assert all(x is not None and x.abs().sum() > 0 for x in g), "after one step the gradient must reach the reduction behind the frozen features"
    print("shapes, zero start, gradient reaches the reduction behind the frozen features after one step: ok")

    # 2. geometry at stride 4 with the depth at stride 16: a painted ground line at y = +2 m, x = 6..40 m lands in the BEV
    #    cells of that line in the ground slice (z = 0), not at y = -2 m
    f = torch.zeros(1, 1, H // 4, W // 4)
    for x in torch.arange(6.0, 40.0, 0.05):
        pc = T @ torch.tensor([x, 2.0, 0.0, 1.0])
        u, v = (K @ (pc[:3] / pc[2]))[:2]
        j, i = int((u + 0.5) / 4), int((v + 0.5) / 4)
        if 0 <= i < f.shape[2] and 0 <= j < f.shape[3]:
            f[0, 0, i, j] = 1.0
    uni = torch.full((1, len(depths), H // 16, W // 16), 1.0 / len(depths))
    lift = LiftPD(grid, depths)
    vol = lift(f, uni, K, T, hw)                               # (1, Z * 1, X, Y), height-major channels
    ground = vol[0, 1]                                         # z = 0.0
    xs, ys = grid.xy_centres()
    on = ground[(xs > 8) & (xs < 30)][:, (ys - 2.0).abs() < 0.3].mean()
    off = ground[(xs > 8) & (xs < 30)][:, (ys + 2.0).abs() < 0.3].mean()
    assert on > 20 * max(off.item(), 1e-6) and on > 0, (on, off)
    print(f"stride-4 ground line lands in its BEV cells: mean on the line {on:.4f}, mirrored side {off:.6f}: ok")

    # 3. res4 kind: frozen (no trainable parameter in the features, BatchNorm stays in eval mode under .train())
    br4 = LaneBranch(grid, depths, c_out=8, oversample=1, pretrained="")
    br4.train()
    assert not any(p.requires_grad for p in br4.features.parameters()) and not br4.features.bn1.training
    assert br4.image_map(img[:1]).shape == (1, 32, H // 4, W // 4)
    print("res4 features frozen, BatchNorm in eval mode under train(): ok")
    assert os.path.exists(DUALBEV_CKPT), f"the pre-trained weights {DUALBEV_CKPT} are not in {os.getcwd()} (README: what the tests need beside the code)"
    Res4Features(DUALBEV_CKPT)
    print("res4 weights load from the DualBEV checkpoint (conv1, bn1, layer1): ok")


if __name__ == "__main__":
    os.chdir(netpath.SITE)                                       # the data folder: the res4 weights are named relative to it
    _selftest()
