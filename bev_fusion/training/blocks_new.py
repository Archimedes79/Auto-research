"""New standard building blocks for the block-exchange study (experiment t2_block_swaps), each a drop-in for one block of
model.BEVFusionModel; written from the published designs, no third-party code.

RadarPillars   point-based radar encoder (PointPillars / RadarPillarNet idea; CRN and RCBEVDet use the same family): a shared
               per-point MLP (properties + sub-cell offset) -> max over the points of a cell (scatter-max) + a count channel
               -> two 3x3 convs -> (B, c_out, NX, NY). Replaces RadarBEV (raster of MEANS + two convs).
SecondTrunk    SECOND backbone + SECONDFPN (the CenterPoint / BEVFusion-lidar BEV encoder): stage 1 at the full grid (stride 1),
               stage 2 at stride 2, both upsampled to the full grid and concatenated. Replaces the DualBEV
               ResNet pyramid + FPN-LSS (BEVTrunk). Not pre-trained here.
CtxAdapter     trainable residual adapter on the cached (frozen, detection-trained) image context features before the lift:
               x + conv3x3(relu(norm(conv3x3(x)))), the last conv zero-initialised, so training starts exactly at the reference
               system (try 6 of t3_improve_loop: are the frozen features the bottleneck?).
Run the self-test: python training/blocks_new.py
"""
import torch
import torch.nn as nn

try:
    from .lift_pd import Grid
    from .radar_bev import HARMONICS, V_MAX, n_properties, properties
except ImportError:                                   # run as a script
    from lift_pd import Grid
    from radar_bev import HARMONICS, V_MAX, n_properties, properties


def _cbr(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True))


class RadarPillars(nn.Module):
    def __init__(self, grid=None, out_channels=32, v_max=V_MAX, harmonics=HARMONICS, c_pfn=48):
        super().__init__()
        self.grid, self.v_max, self.harmonics, self.c_pfn = grid or Grid(), v_max, harmonics, c_pfn
        cin = n_properties(harmonics) + 2
        self.pfn = nn.Sequential(nn.Linear(cin, c_pfn), nn.BatchNorm1d(c_pfn), nn.ReLU(True),
                                 nn.Linear(c_pfn, c_pfn), nn.BatchNorm1d(c_pfn), nn.ReLU(True))
        self.net = nn.Sequential(nn.Conv2d(c_pfn + 1, out_channels, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU())

    def pillars(self, det, B):
        """-> (B, c_pfn + 1, NX, NY): per-cell max of the point features, and log(1 + count); empty cells are zero."""
        g = self.grid
        fx, fy = (det["x"] - g.x_min) / g.res, (det["y"] - g.y_min) / g.res
        ix, iy = fx.floor().long(), fy.floor().long()
        inside = (ix >= 0) & (ix < g.nx) & (iy >= 0) & (iy < g.ny)
        dev = det["x"].device
        if not bool(inside.any()):
            return torch.zeros(B, self.c_pfn + 1, g.nx, g.ny, device=dev)
        d = {k: v[inside] for k, v in det.items()}
        feats = torch.cat([properties(d, self.v_max, self.harmonics), (fx - ix - 0.5)[inside][:, None],
                           (fy - iy - 0.5)[inside][:, None]], -1)
        h = self.pfn(feats) if feats.shape[0] > 1 else self.pfn.eval()(feats)
        cell = (d["b"] * g.nx + ix[inside]) * g.ny + iy[inside]
        out = torch.zeros(B * g.nx * g.ny, self.c_pfn, device=dev, dtype=h.dtype)
        out.scatter_reduce_(0, cell[:, None].expand_as(h), h, reduce="amax", include_self=False)
        cnt = torch.zeros(B * g.nx * g.ny, 1, device=dev, dtype=h.dtype)
        cnt.index_add_(0, cell, torch.ones_like(cell, dtype=h.dtype)[:, None])
        x = torch.cat([out, torch.log1p(cnt)], 1)
        return x.view(B, g.nx, g.ny, -1).permute(0, 3, 1, 2).contiguous()

    def forward(self, det, B):
        return self.net(self.pillars(det, B))


class SecondTrunk(nn.Module):
    """SECOND (layer_nums, channels, strides 1 and 2) + SECONDFPN (up to the full grid, concat). features(x) -> (B, out_channels, X, Y)."""

    def __init__(self, cin=96, channels=(128, 256), layer_nums=(3, 3), fpn_out=(128, 128)):
        super().__init__()
        stages, cur = [], cin
        for i, (c, n) in enumerate(zip(channels, layer_nums)):
            layers = [_cbr(cur, c, 1 if i == 0 else 2)] + [_cbr(c, c) for _ in range(n - 1)]
            stages.append(nn.Sequential(*layers))
            cur = c
        self.stages = nn.ModuleList(stages)
        self.ups = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels[0], fpn_out[0], 1, bias=False), nn.BatchNorm2d(fpn_out[0]), nn.ReLU(True)),
            nn.Sequential(nn.ConvTranspose2d(channels[1], fpn_out[1], 2, 2, bias=False), nn.BatchNorm2d(fpn_out[1]),
                          nn.ReLU(True))])
        self.out_channels = sum(fpn_out)

    def features(self, x):
        h, w = x.shape[-2:]
        f1 = self.stages[0](x)
        f2 = self.stages[1](f1)
        u2 = self.ups[1](f2)
        if u2.shape[-2:] != (h, w):                       # odd grid sizes: the transposed conv rounds down
            u2 = nn.functional.interpolate(u2, size=(h, w), mode="bilinear", align_corners=False)
        return torch.cat([self.ups[0](f1), u2], 1)

    forward = features


class CtxAdapter(nn.Module):
    def __init__(self, c=80, hidden=96):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(c, hidden, 3, padding=1, bias=False), nn.GroupNorm(8, hidden), nn.ReLU(True),
                                  nn.Conv2d(hidden, c, 3, padding=1))
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x):                                  # (N, c, H, W)
        return x + self.body(x)


def _selftest():
    torch.manual_seed(0)
    g = Grid()
    n = 400
    det = {"b": torch.randint(0, 2, (n,)), "x": torch.rand(n) * 60, "y": torch.rand(n) * 52 - 26,
           "r": torch.rand(n) * 60, "psi": torch.rand(n) * 6 - 3, "vr": torch.randn(n) * 5, "rcs": torch.randn(n) * 5,
           "dt": torch.rand(n) * 0.5}
    m = RadarPillars(g, 32).train()
    out = m(det, 2)
    assert out.shape == (2, 32, g.nx, g.ny) and torch.isfinite(out).all()
    # the max over the points of a cell equals a brute-force max of the per-point features
    p = m.pillars(det, 2)
    with torch.no_grad():
        mm = RadarPillars(g, 32).eval()
        mm.load_state_dict(m.state_dict())
        pe = mm.pillars(det, 2)
        fx, fy = (det["x"] - g.x_min) / g.res, (det["y"] - g.y_min) / g.res
        ix, iy = fx.floor().long(), fy.floor().long()
        feats = torch.cat([properties(det, m.v_max, m.harmonics), (fx - ix - 0.5)[:, None], (fy - iy - 0.5)[:, None]], -1)
        h = mm.pfn(feats)
        for i in range(0, n, 37):
            same = (det["b"] == det["b"][i]) & (ix == ix[i]) & (iy == iy[i])
            assert torch.allclose(pe[det["b"][i], :-1, ix[i], iy[i]], h[same].max(0).values, atol=1e-5)
            assert abs(float(pe[det["b"][i], -1, ix[i], iy[i]]) - float(torch.log1p(same.sum().float()))) < 1e-5
        assert float(pe[:, :-1].abs().sum()) > 0 and float((pe[:, -1] == 0).float().mean()) > 0.9, "empty cells are zero"
    out.mean().backward()
    assert all(q.grad is not None for q in m.parameters()), "gradient reaches every radar-pillar parameter"
    assert torch.isfinite(m({k: v[:0] for k, v in det.items()}, 2)).all(), "no detections"
    t = SecondTrunk(cin=96).train()
    x = torch.randn(2, 96, g.nx, g.ny, requires_grad=True)
    y = t.features(x)
    assert y.shape == (2, 256, g.nx, g.ny)
    y.mean().backward()
    assert x.grad is not None and all(q.grad is not None for q in t.parameters())
    a = CtxAdapter(80)
    xa = torch.randn(3, 80, 16, 44)
    assert torch.equal(a(xa), xa), "zero-initialised adapter is the identity"
    a(xa).sum().backward()
    assert a.body[-1].weight.grad is not None and a.body[0].weight.grad is not None
    print(f"blocks_new selftest ok: RadarPillars (scatter-max = brute force, count channel, empty input, gradients; "
          f"{sum(q.numel() for q in m.parameters()) / 1e3:.1f} k parameters), SecondTrunk (output {tuple(y.shape)}, "
          f"{sum(q.numel() for q in t.parameters()) / 1e6:.2f} M parameters), CtxAdapter (identity at init, gradients)")


if __name__ == "__main__":
    _selftest()
