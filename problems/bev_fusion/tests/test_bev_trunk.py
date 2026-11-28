"""Tests of net/model/bev_trunk.py (its former self-test).

    python -P -m bev_fusion.tests.test_bev_trunk      (from the problem folder, with its module path)
"""
import torch

from bev_fusion.net.model.bev_trunk import BEVTrunk


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def _selftest():
    torch.manual_seed(0)
    x = torch.randn(2, 80, 120, 104)
    trunk = BEVTrunk(cin=80, n_classes=14).eval()
    with torch.no_grad():
        feats = trunk.img_bev_encoder_backbone(x)
        assert [tuple(f.shape[1:]) for f in feats] == [(160, 60, 52), (320, 30, 26), (640, 15, 13)]
        y = trunk(x)
    assert y.shape == (2, 14, 120, 104)
    enc = n_params(trunk.img_bev_encoder_backbone) + n_params(trunk.img_bev_encoder_neck)
    stats = sum(v.numel() for k, v in trunk.state_dict().items()
                if k.startswith(("img_bev_encoder_backbone.", "img_bev_encoder_neck.")) and "num_batches" not in k)
    assert stats == 26668896, stats          # parameters + BatchNorm statistics = the tensors of DualBEV's BEV encoder
    # the narrow variants keep the output shape; widths scale the parameters roughly with w^2
    for kw in ({"width": 0.5}, {"width": 0.25}):
        t = BEVTrunk(cin=48, n_classes=14, **kw).eval()
        with torch.no_grad():
            assert t(torch.randn(1, 48, 120, 104)).shape == (1, 14, 120, 104)
    # the stride-1 detail path: output shape unchanged, the pre-trained tensors unchanged, the head sees the extra channels
    for kw in ({"width": 0.5, "detail": 32}, {"width": 1.0, "detail": 32}):
        t = BEVTrunk(cin=48, n_classes=14, **kw).eval()
        with torch.no_grad():
            assert t(torch.randn(1, 48, 120, 104)).shape == (1, 14, 120, 104)
        assert t.head.net[0].in_channels == t.img_bev_encoder_neck.out_channels + 32
    td = BEVTrunk(cin=80, n_classes=14, detail=32).eval()
    stats_d = sum(v.numel() for k, v in td.state_dict().items()
                  if k.startswith(("img_bev_encoder_backbone.", "img_bev_encoder_neck.")) and "num_batches" not in k)
    assert stats_d == 26668896, "the detail path must not change the pre-trained encoder tensors"
    td.train()
    td(x).mean().backward()
    assert all(p.grad is not None for p in td.parameters()), "gradient reaches the detail path"
    # a detail-path signal reaches the output at full resolution: changing ONE input cell changes the output within +-1 cell
    # of it by far more than at a cell 6 cells away (no pyramid path is needed for that)
    td.eval()
    with torch.no_grad():
        base = td(torch.zeros(1, 80, 120, 104))
        pert = torch.zeros(1, 80, 120, 104)
        pert[0, :, 60, 50] = 1.0
        d = (td(pert) - base).abs().sum(1)[0]
    assert d[59:62, 49:52].max() > 5 * d[66, 56].clamp(min=1e-12), "the detail path is local"
    # gradient reaches the input and every parameter
    trunk.train()
    trunk(x).mean().backward()
    assert all(p.grad is not None for p in trunk.parameters())
    print(f"bev_trunk selftest ok: shapes of the three stages, output (2, 14, 120, 104), {enc / 1e6:.2f} M encoder "
          f"parameters = DualBEV's, narrow variants, gradients")


if __name__ == "__main__":
    _selftest()
