"""Tests of net/model/camera_encoder.py (its former self-test).

    python tests/test_camera_encoder.py      (from the problem folder, with its module path)
"""
import torch

from net.model.camera_encoder import FrontCameraEncoder
from net.prep.images import camera_vector, nuscenes_front_ida


def _selftest():
    torch.manual_seed(0)
    enc = FrontCameraEncoder().eval()
    n_backbone = sum(p.numel() for p in enc.img_backbone.parameters())
    n_heads = sum(p.numel() for m in (enc.img_neck, enc.depth_net) for p in m.parameters())
    # key names the DualBEV checkpoint should provide (derived from its code)
    keys = enc.state_dict().keys()
    for k in ("img_backbone.layer3.0.downsample.0.weight", "img_neck.lateral_convs.1.conv.weight",
              "img_neck.fpn_convs.0.conv.weight", "depth_net.bn.running_mean", "depth_net.depth_mlp.fc1.weight",
              "depth_net.depth_se.conv_reduce.weight", "depth_net.depth_conv.3.aspp2.atrous_conv.weight",
              "depth_net.depth_conv.4.weight", "depth_net.context_conv.weight"):
        assert k in keys, k
    # image transform: 1600 x 900 -> 704 x 396, 140 rows cropped at the top
    ida = nuscenes_front_ida()
    assert abs(ida[0, 0] - 0.44) < 1e-6 and ida[1, 2] == -140
    # front camera of a synthetic rig (camera 1.5 m high, looking along +x)
    K = torch.tensor([[1266.0, 0, 799.5], [0, 1266.0, 449.5], [0, 0, 1]])
    R = torch.tensor([[0.0, -1, 0], [0, 0, -1], [1, 0, 0]])
    T = torch.eye(4)
    T[:3, :3], T[:3, 3] = R, -R @ torch.tensor([0.0, 0, 1.5])
    s2e = torch.linalg.inv(T)
    B = 2
    cam = camera_vector(K.expand(B, 3, 3), ida[:2, :2].expand(B, 2, 2), ida[:2, 2].expand(B, 2), s2e.expand(B, 4, 4))
    assert cam.shape == (B, 27) and cam[0, 0] == 1266.0 and cam[0, 15 + 3] == s2e[0, 3]
    img = torch.randn(B, 3, 256, 704)
    with torch.no_grad():
        prob, ctx = enc.image_features(img, cam)
        assert prob.shape == (B, 118, 16, 44) and ctx.shape == (B, 80, 16, 44)
        assert torch.allclose(prob.sum(1), torch.ones(B, 16, 44), atol=1e-5)
        # one camera alone gives the same result as inside a batch (eval mode: no cross-sample statistics)
        p1, c1 = enc.image_features(img[:1], cam[:1])
        assert torch.allclose(p1, prob[:1], atol=1e-5) and torch.allclose(c1, ctx[:1], atol=1e-4)
        bev = enc(img, K, T, ida, cam)
    assert bev.shape == (B, 128, 120, 104)
    # frozen heads: only the BEV side trains
    enc.freeze_camera_heads()
    enc.train()
    trainable = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    out = enc(img, K, T, ida, cam)
    out.mean().backward()
    assert enc.out_conv[0].weight.grad is not None and enc.depth_net.depth_conv[4].weight.grad is None
    assert not enc.depth_net.training
    print(f"camera_encoder selftest ok: backbone {n_backbone / 1e6:.1f} M, neck + DepthNet {n_heads / 1e6:.1f} M "
          f"parameters (frozen), trainable {trainable / 1e3:.0f} k; BEV {tuple(bev.shape)}")


if __name__ == "__main__":
    _selftest()
