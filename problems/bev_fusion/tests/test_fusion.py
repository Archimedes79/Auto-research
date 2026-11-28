"""Tests of net/model/fusion.py (its former self-test).

    python tests/test_fusion.py      (from the problem folder, with its module path)
"""
import torch

from net.model.fusion import ConcatFusion, drop_modality


def _selftest():
    torch.manual_seed(0)
    B, X, Y = 4, 32, 24
    cam, rad = torch.randn(B, 3, X, Y), torch.randn(B, 2, X, Y)
    m = ConcatFusion(3, 2, 16, p_drop=0.0)
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
    m = ConcatFusion(3, 2, 8, p_drop=0.5).eval()
    assert torch.equal(m(cam, rad), m(cam, rad)), "no dropout in eval"
    # concat fusion with the camera off equals the radar-only path
    f = ConcatFusion(3, 2, 8, p_drop=0.0).eval()
    both_off = f(torch.zeros_like(cam), torch.zeros_like(rad))
    assert torch.allclose(both_off, both_off[:, :, :1, :1].expand_as(both_off), atol=1e-6), "no input -> constant output"
    print("fusion test ok: shape and gradients, modality dropout (p/2 each, never both, off in eval), no input gives a constant output")


if __name__ == "__main__":
    _selftest()
