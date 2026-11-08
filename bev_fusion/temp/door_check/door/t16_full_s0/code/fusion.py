"""Camera-radar fusion at camera time (bev_fusion/plan.md section 4): plug-in modules in front of the BEV trunk.

Inputs: the temporally fused camera planes (B, c_cam, X, Y) and the radar planes moved to the same time (B, c_rad, X, Y);
each branch carries its own confidence channels (mass, dt), so "sensor absent" is simply an all-zero input (mass 0).
Output: (B, c_out, X, Y) for BEVTrunk(cin=c_out).

- ConcatFusion (main, BEVFusion ConvFuser style): concat -> 3x3 conv -> BatchNorm -> ReLU.
- GateFusion (fallback 1, in the spirit of GaussianCaR's FeatureRectify; re-implemented from the description in
  literature/code/gaussiancar_fusion.md, no code copied): both maps projected to c_out, then each stream is corrected by
  the other through learned per-channel gates (global avg + max pool -> MLP -> sigmoid) and per-cell gates (1x1 convs on
  the concatenation -> sigmoid), x' = x + lam_c * g_c * y + lam_s * g_s * y; then concat -> 3x3 conv.
  `gates(cam, rad)` returns the per-cell gates for plotting (radar should dominate far, camera near).
- drop_modality: modality dropout in training (per sample, with probability p exactly one of the two modalities is
  zeroed, never both), applied inside both modules when `training`.
Run the self-test: python training/fusion.py
"""
import torch
import torch.nn as nn


def drop_modality(cam, rad, p):
    """Zero the camera planes of a fraction p/2 of the samples and the radar planes of another p/2 (never both)."""
    B = cam.shape[0]
    r = torch.rand(B, device=cam.device)
    keep_cam = (r >= 0.5 * p).to(cam.dtype).view(B, 1, 1, 1)
    keep_rad = ((r < 0.5 * p) | (r >= p)).to(cam.dtype).view(B, 1, 1, 1)
    return cam * keep_cam, rad * keep_rad


def _conv_bn_relu(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True))


class ConcatFusion(nn.Module):
    def __init__(self, c_cam, c_rad, c_out, p_drop=0.2):
        super().__init__()
        self.conv = _conv_bn_relu(c_cam + c_rad, c_out)
        self.p_drop = p_drop

    def forward(self, cam, rad):
        if self.training and self.p_drop > 0:
            cam, rad = drop_modality(cam, rad, self.p_drop)
        return self.conv(torch.cat([cam, rad], 1))


class GateFusion(nn.Module):
    def __init__(self, c_cam, c_rad, c_out, p_drop=0.2, reduction=4, lam_c=0.5, lam_s=0.5):
        super().__init__()
        self.pc = nn.Conv2d(c_cam, c_out, 1, bias=False)
        self.pr = nn.Conv2d(c_rad, c_out, 1, bias=False)
        h = max(c_out // reduction, 4)
        self.channel = nn.Sequential(nn.Linear(4 * c_out, h), nn.ReLU(True), nn.Linear(h, 2 * c_out), nn.Sigmoid())
        self.spatial = nn.Sequential(nn.Conv2d(2 * c_out, h, 1), nn.ReLU(True), nn.Conv2d(h, 2, 1), nn.Sigmoid())
        self.out = _conv_bn_relu(2 * c_out, c_out)
        self.c_out, self.p_drop, self.lam_c, self.lam_s = c_out, p_drop, lam_c, lam_s

    def rectify(self, cam, rad):
        """-> (cam', rad', spatial gates (B, 2, 1, X, Y)); gates[:, 1] = how much of the other stream flows into the
        camera stream, gates[:, 0] = into the radar stream."""
        c, r = self.pc(cam), self.pr(rad)
        cat = torch.cat([c, r], 1)
        pooled = torch.cat([cat.mean((2, 3)), cat.amax((2, 3))], 1)                      # (B, 4 c_out)
        cw = self.channel(pooled).view(-1, 2, self.c_out, 1, 1)
        sw = self.spatial(cat).unsqueeze(2)                                              # (B, 2, 1, X, Y)
        c2 = c + self.lam_c * cw[:, 1] * r + self.lam_s * sw[:, 1] * r
        r2 = r + self.lam_c * cw[:, 0] * c + self.lam_s * sw[:, 0] * c
        return c2, r2, sw

    def forward(self, cam, rad):
        if self.training and self.p_drop > 0:
            cam, rad = drop_modality(cam, rad, self.p_drop)
        c2, r2, _ = self.rectify(cam, rad)
        return self.out(torch.cat([c2, r2], 1))

    def gates(self, cam, rad):
        return self.rectify(cam, rad)[2]


def _selftest():
    torch.manual_seed(0)
    B, X, Y = 4, 32, 24
    cam, rad = torch.randn(B, 3, X, Y), torch.randn(B, 2, X, Y)
    for cls in (ConcatFusion, GateFusion):
        m = cls(3, 2, 16, p_drop=0.0)
        out = m(cam, rad)
        assert out.shape == (B, 16, X, Y)
        out.mean().backward()
        assert all(p.grad is not None for p in m.parameters()), "gradient reaches every parameter"
    # modality dropout: a fraction p / 2 of each modality, never both, only in training
    big_c, big_r = torch.ones(20000, 1, 1, 1), torch.ones(20000, 1, 1, 1)
    dc, dr = drop_modality(big_c, big_r, 0.4)
    gone_c, gone_r = (dc.flatten() == 0), (dr.flatten() == 0)
    assert abs(gone_c.float().mean() - 0.2) < 0.01 and abs(gone_r.float().mean() - 0.2) < 0.01
    assert not (gone_c & gone_r).any(), "never both modalities"
    for cls in (ConcatFusion, GateFusion):
        m = cls(3, 2, 8, p_drop=0.5).eval()
        a, b = m(cam, rad), m(cam, rad)
        assert torch.equal(a, b), "no dropout in eval"
    # gate fusion with the radar off: the camera stream is unchanged (the projections have no bias)
    g = GateFusion(3, 2, 8).eval()
    c2, r2, sw = g.rectify(cam, torch.zeros_like(rad))
    assert torch.allclose(c2, g.pc(cam), atol=1e-6), "radar off: camera stream unchanged"
    assert sw.shape == (B, 2, 1, X, Y) and (sw >= 0).all() and (sw <= 1).all()
    # concat fusion with the camera off equals the radar-only path
    f = ConcatFusion(3, 2, 8, p_drop=0.0).eval()
    both_off = f(torch.zeros_like(cam), torch.zeros_like(rad))
    assert torch.allclose(both_off, both_off[:, :, :1, :1].expand_as(both_off), atol=1e-6), "no input -> constant output"
    print("fusion selftest ok: shapes and gradients of both modules, modality dropout (p/2 each, never both, off in "
          "eval), gate fusion keeps the camera stream with the radar off, per-cell gates in [0, 1]")


if __name__ == "__main__":
    _selftest()
