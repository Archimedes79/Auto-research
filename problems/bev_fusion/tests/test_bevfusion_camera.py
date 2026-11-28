"""Tests of net/model/bevfusion_camera.py (its former self-test).

    python tests/test_bevfusion_camera.py      (from the problem folder, with its module path)
"""
import os

import torch
import torch.nn as nn

from net.model.bevfusion_camera import BEVFusionImageSide, CKPT, PatchMerging, ShiftWindowMSA, SwinT, window_partition, window_reverse
from net.prep.geometry import DEPTHS


def _selftest():
    torch.manual_seed(0)
    # shapes and the window machinery on a random model
    m = BEVFusionImageSide().eval()
    x = torch.randn(1, 3, 256, 704)
    with torch.no_grad():
        outs = m.backbone(x)
        prob, ctx = m.image_features(x)
    assert [tuple(o.shape[1:]) for o in outs] == [(192, 32, 88), (384, 16, 44), (768, 8, 22)], [o.shape for o in outs]
    assert prob.shape == (1, 118, 32, 88) and ctx.shape == (1, 80, 32, 88) and torch.allclose(prob.sum(1), torch.ones(1, 32, 88), atol=1e-5)
    # window partition / reverse are inverse; shifted attention mask blocks cross-region pairs (values 0 / -100, as mmdet)
    t = torch.randn(2, 14, 21, 5)
    assert torch.equal(window_reverse(window_partition(t, 7), 7, 14, 21), t)
    msk = ShiftWindowMSA(8, 2, 7, 3).mask(14, 14, "cpu")
    assert msk.shape == (4, 49, 49) and set(msk.unique().tolist()) <= {0.0, -100.0} and (msk[0] == 0).all() and (msk[3] == -100).any()
    # patch merging channel order = nn.Unfold's (c-major)
    pm = PatchMerging(3, 6)
    xx = torch.arange(1 * 4 * 4 * 3, dtype=torch.float32).view(1, 16, 3)
    pm.norm = nn.Identity()
    pm.reduction = nn.Identity()
    out, hw = pm(xx, (4, 4))
    assert hw == (2, 2) and out.shape == (1, 4, 12)
    img = xx.view(1, 4, 4, 3)
    assert out[0, 0, 0] == img[0, 0, 0, 0] and out[0, 0, 1] == img[0, 0, 1, 0] and out[0, 0, 2] == img[0, 1, 0, 0] and out[0, 0, 4] == img[0, 0, 0, 1], "unfold order"
    # independent reference: torchvision's swin_t with random weights converted into our key layout gives the same stage outputs
    # (window attention, shifts, masks, padding, relative position bias, merging order are checked against code we did not write)
    import torchvision
    tv = torchvision.models.swin_t(weights=None).eval()
    for p_ in tv.parameters():
        p_.data.normal_(0, 0.3)
    mine = SwinT().eval()
    for i in mine.out_indices:
        setattr(mine, f"norm{i}", nn.Identity())
    new = {"patch_embed.projection.weight": tv.features[0][0].weight, "patch_embed.projection.bias": tv.features[0][0].bias,
           "patch_embed.norm.weight": tv.features[0][2].weight, "patch_embed.norm.bias": tv.features[0][2].bias}
    for i, st in enumerate(mine.stages):
        for j in range(len(st.blocks)):
            b = tv.features[1 + 2 * i][j]
            pre = f"stages.{i}.blocks.{j}."
            new.update({pre + "norm1.weight": b.norm1.weight, pre + "norm1.bias": b.norm1.bias, pre + "norm2.weight": b.norm2.weight,
                        pre + "norm2.bias": b.norm2.bias, pre + "attn.w_msa.qkv.weight": b.attn.qkv.weight,
                        pre + "attn.w_msa.qkv.bias": b.attn.qkv.bias, pre + "attn.w_msa.proj.weight": b.attn.proj.weight,
                        pre + "attn.w_msa.proj.bias": b.attn.proj.bias,
                        pre + "attn.w_msa.relative_position_bias_table": b.attn.relative_position_bias_table,
                        pre + "ffn.layers.0.0.weight": b.mlp[0].weight, pre + "ffn.layers.0.0.bias": b.mlp[0].bias,
                        pre + "ffn.layers.1.weight": b.mlp[3].weight, pre + "ffn.layers.1.bias": b.mlp[3].bias})
        if st.downsample is not None:
            m_ = tv.features[2 + 2 * i]
            C = m_.reduction.weight.shape[1] // 4
            # torchvision concatenates [x(0,0), x(1,0), x(0,1), x(1,1)] (index j * C + c); nn.Unfold gives c * 4 + (kh * 2 + kw)
            perm = torch.tensor([(kh + 2 * kw) * C + c for c in range(C) for kh in range(2) for kw in range(2)])
            new[f"stages.{i}.downsample.norm.weight"] = m_.norm.weight[perm]
            new[f"stages.{i}.downsample.norm.bias"] = m_.norm.bias[perm]
            new[f"stages.{i}.downsample.reduction.weight"] = m_.reduction.weight[:, perm]
    res = mine.load_state_dict({k: v.detach().clone() for k, v in new.items()}, strict=False)
    assert not res.unexpected_keys and all(k.endswith("relative_position_index") for k in res.missing_keys), res
    with torch.no_grad():
        ref = [tv.features[:4](x), tv.features[:6](x), tv.features[:8](x)]
        got = mine(x)
    for r, g in zip(ref, got):
        err = (r.permute(0, 3, 1, 2) - g).abs().max() / r.abs().max()
        assert err < 1e-4, f"swin vs torchvision: relative error {float(err):.2e}"
    # the checkpoint: every tensor of the three parts is used, relative position indices equal ours, frustum depths 1.0 + 0.5 i
    if os.path.exists(CKPT):
        rep = BEVFusionImageSide().load_pretrained(CKPT)
        assert not rep["unexpected"], rep["unexpected"][:5]
        miss = [k for k in rep["missing"] if not k.endswith("relative_position_index")]
        assert not miss, miss[:5]
        assert torch.allclose(rep["frustum_depths"], DEPTHS), "frustum depths"
        sd = torch.load(CKPT, map_location="cpu", weights_only=True)["state_dict"]
        ours = SwinT().state_dict()
        for k, v in sd.items():
            if k.startswith("encoders.camera.backbone.") and k.endswith("relative_position_index"):
                assert torch.equal(ours[k[len("encoders.camera.backbone."):]], v), k
        n = sum(p.numel() for p in BEVFusionImageSide().parameters())
        print(f"bevfusion_camera selftest ok: shapes (192 / 384 / 768 ch at strides 8 / 16 / 32, probability + context at 32 x 88), window "
              f"partition, shift mask, unfold channel order, Swin-T equal to torchvision's swin_t on converted random weights; checkpoint: {rep['loaded']} tensors loaded, none unexpected, none missing "
              f"(relative position indices equal the checkpoint's), frustum depths 1.0 + 0.5 i, {n / 1e6:.1f} M parameters")
    else:
        print("bevfusion_camera selftest ok (no checkpoint file): shapes, window partition, shift mask, unfold order")


if __name__ == "__main__":
    _selftest()
