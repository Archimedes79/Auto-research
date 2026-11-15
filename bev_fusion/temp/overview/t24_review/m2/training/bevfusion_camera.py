"""Image side of BEVFusion's camera-only map-segmentation model (2205.13542, Apache-2.0 code; weights models/pretrained/bevfusion/camera-only-seg.pth):
Swin-T -> GeneralizedLSSFPN -> a 1 x 1 conv that gives 118 depth logits + 80 context channels at stride 8, re-implemented in plain PyTorch
(no mmcv / mmdet3d) with mmdet's Swin semantics and key names, so the checkpoint loads without renaming (`load_pretrained`).

    image (B, 3, 256, 704), ImageNet-normalised (BEVFusion test pipeline: resize by 0.48 with PIL bicubic -> 768 x 432, crop (32, 176, 736, 432))
      -> Swin-T (embed 96, depths 2 2 6 2, heads 3 6 12 24, window 7, patch 4; outputs after stages 1, 2, 3 normalised: 192 / 384 / 768 ch at
         strides 8 / 16 / 32)
      -> GeneralizedLSSFPN: top-down by upsample + CONCAT + 1 x 1 conv + 3 x 3 conv (BN, ReLU), 256 ch; level 0 = stride 8 (32 x 88 cells)
      -> depthnet: ONE 1 x 1 conv 256 -> 198: softmax over the first 118 = depth bins d_i = 1.0 + 0.5 i m (the frustum buffer of the
         checkpoint: LEFT edges), the last 80 = context
Same call as the other encoders: `image_features(img) -> (prob (B, 118, 32, 88), ctx (B, 80, 32, 88))`. Positions: BEVFusion's frustum puts
feature cell j at pixel x = j (W - 1) / (w - 1) (and y alike), not at the cell centre (`frustum_pixels`): half a cell of 4 px against our lift's
centre convention. The model was trained for BEV map segmentation with six cameras; its depth distribution comes from that task loss.
The BEV part of the checkpoint (vtransform.downsample, decoder, heads) is not used here.
Run the self-test: python training/bevfusion_camera.py    (probe on real images: temp/checks/probe_bevfusion_depth.py)
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

CKPT = "models/pretrained/bevfusion/camera-only-seg.pth"
IMG_MEAN, IMG_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)             # on 0..1
DEPTHS = 1.0 + 0.5 * torch.arange(118, dtype=torch.float32)                  # frustum: bin i at 1.0 + 0.5 i (left edge)


def bevfusion_ida(scale=0.48, crop=(32, 176)):
    """3 x 3 pixel transform original -> augmented image (test pipeline: scale, crop x0 / y0)."""
    return torch.tensor([[scale, 0, -crop[0]], [0, scale, -crop[1]], [0, 0, 1]], dtype=torch.float32)


def frustum_pixels(fh=32, fw=88, h=256, w=704):
    """augmented-image pixel (x, y) of every feature cell in BEVFusion's frustum: linspace(0, W - 1, fW) x linspace(0, H - 1, fH)."""
    ys = torch.linspace(0, h - 1, fh)
    xs = torch.linspace(0, w - 1, fw)
    return xs, ys


def load_image(path, scale=0.48, crop=(32, 176, 736, 432)):
    """BEVFusion test-time image: PIL bicubic resize, crop, ImageNet normalisation -> (3, 256, 704)."""
    import numpy as np
    from PIL import Image
    img = Image.open(path).convert("RGB")
    img = img.resize((int(img.width * scale), int(img.height * scale)), Image.BICUBIC).crop(crop)
    x = torch.from_numpy(np.asarray(img).copy()).float().div(255).permute(2, 0, 1)
    return (x - torch.tensor(IMG_MEAN).view(3, 1, 1)) / torch.tensor(IMG_STD).view(3, 1, 1)


# ------------------------------------------------------------------------------------------------ Swin-T (mmdet semantics)
class WindowMSA(nn.Module):
    def __init__(self, dim, heads, window=7):
        super().__init__()
        self.dim, self.heads, self.window, self.scale = dim, heads, window, (dim // heads) ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window - 1) ** 2, heads))
        coords = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window), indexing="ij")).flatten(1)   # (2, N)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0) + (window - 1)
        self.register_buffer("relative_position_index", rel[..., 0] * (2 * window - 1) + rel[..., 1])
        self.qkv, self.proj = nn.Linear(dim, dim * 3), nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = (attn.view(B // nW, nW, self.heads, N, N) + mask.unsqueeze(1).unsqueeze(0)).view(-1, self.heads, N, N)
        return self.proj((attn.softmax(-1) @ v).transpose(1, 2).reshape(B, N, C))


def window_partition(x, w):
    B, H, W, C = x.shape
    return x.view(B, H // w, w, W // w, w, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, w, w, C)


def window_reverse(win, w, H, W):
    B = int(win.shape[0] / (H * W / w / w))
    return win.view(B, H // w, W // w, w, w, -1).permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)


class ShiftWindowMSA(nn.Module):
    def __init__(self, dim, heads, window, shift):
        super().__init__()
        self.window, self.shift = window, shift
        self.w_msa = WindowMSA(dim, heads, window)
        self._mask = {}

    def mask(self, Hp, Wp, device):
        key = (Hp, Wp, str(device))
        if key not in self._mask:
            w, s = self.window, self.shift
            img = torch.zeros(1, Hp, Wp, 1)
            cnt = 0
            for h in (slice(0, -w), slice(-w, -s), slice(-s, None)):
                for ww in (slice(0, -w), slice(-w, -s), slice(-s, None)):
                    img[:, h, ww, :] = cnt
                    cnt += 1
            mw = window_partition(img, w).view(-1, w * w)
            m = mw.unsqueeze(1) - mw.unsqueeze(2)
            self._mask[key] = m.masked_fill(m != 0, -100.0).masked_fill(m == 0, 0.0).to(device)
        return self._mask[key]

    def forward(self, x, hw):
        B, L, C = x.shape
        H, W = hw
        x = x.view(B, H, W, C)
        w = self.window
        x = F.pad(x, (0, 0, 0, (w - W % w) % w, 0, (w - H % w) % w))
        Hp, Wp = x.shape[1:3]
        if self.shift > 0:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))
            mask = self.mask(Hp, Wp, x.device)
        else:
            mask = None
        out = self.w_msa(window_partition(x, w).view(-1, w * w, C), mask).view(-1, w, w, C)
        x = window_reverse(out, w, Hp, Wp)
        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))
        return x[:, :H, :W, :].contiguous().view(B, H * W, C)


class FFN(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.layers = nn.Sequential(nn.Sequential(nn.Linear(dim, hidden), nn.GELU()), nn.Linear(hidden, dim))

    def forward(self, x):
        return self.layers(x)


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, window, shift, ratio=4):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = ShiftWindowMSA(dim, heads, window, shift)
        self.ffn = FFN(dim, dim * ratio)

    def forward(self, x, hw):
        x = x + self.attn(self.norm1(x), hw)
        return x + self.ffn(self.norm2(x))


class PatchMerging(nn.Module):
    """mmdet's: nn.Unfold(2, stride 2) (channel order c-major: c * 4 + (kh * 2 + kw)), LayerNorm(4C), Linear(4C -> 2C, no bias)."""

    def __init__(self, cin, cout):
        super().__init__()
        self.norm, self.reduction = nn.LayerNorm(4 * cin), nn.Linear(4 * cin, cout, bias=False)

    def forward(self, x, hw):
        B, L, C = x.shape
        H, W = hw
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = F.pad(x, (0, W % 2, 0, H % 2))
        H2, W2 = x.shape[-2] // 2, x.shape[-1] // 2
        x = F.unfold(x, 2, stride=2).transpose(1, 2)
        return self.reduction(self.norm(x)), (H2, W2)


class SwinStage(nn.Module):
    def __init__(self, dim, heads, depth, window, downsample):
        super().__init__()
        self.blocks = nn.ModuleList([SwinBlock(dim, heads, window, 0 if i % 2 == 0 else window // 2) for i in range(depth)])
        self.downsample = PatchMerging(dim, dim * 2) if downsample else None

    def forward(self, x, hw):
        for b in self.blocks:
            x = b(x, hw)
        if self.downsample is None:
            return x, hw, x, hw
        xd, hwd = self.downsample(x, hw)
        return xd, hwd, x, hw


class PatchEmbed(nn.Module):
    def __init__(self, dim=96):
        super().__init__()
        self.projection, self.norm = nn.Conv2d(3, dim, 4, stride=4), nn.LayerNorm(dim)

    def forward(self, x):
        x = self.projection(x)
        hw = x.shape[-2:]
        return self.norm(x.flatten(2).transpose(1, 2)), (hw[0], hw[1])


class SwinT(nn.Module):
    def __init__(self, dim=96, depths=(2, 2, 6, 2), heads=(3, 6, 12, 24), window=7, out_indices=(1, 2, 3)):
        super().__init__()
        self.out_indices = out_indices
        self.patch_embed = PatchEmbed(dim)
        self.stages = nn.ModuleList([SwinStage(dim * 2 ** i, heads[i], depths[i], window, i < len(depths) - 1) for i in range(len(depths))])
        for i in out_indices:
            self.add_module(f"norm{i}", nn.LayerNorm(dim * 2 ** i))

    def forward(self, x):
        x, hw = self.patch_embed(x)
        outs = []
        for i, st in enumerate(self.stages):
            x, hw, out, ohw = st(x, hw)
            if i in self.out_indices:
                o = getattr(self, f"norm{i}")(out)
                outs.append(o.view(-1, ohw[0], ohw[1], o.shape[-1]).permute(0, 3, 1, 2).contiguous())
        return outs


# ------------------------------------------------------------------------------------------------ neck, depth head
class ConvBNReLU(nn.Module):
    def __init__(self, cin, cout, k):
        super().__init__()
        self.conv, self.bn = nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class GeneralizedLSSFPN(nn.Module):
    """BEVFusion's neck: per level upsample the coarser output, CONCATENATE it with the lateral map, 1 x 1 conv, 3 x 3 conv; two outputs
    (strides 8 and 16) from three inputs (192 / 384 / 768 ch)."""

    def __init__(self, chans=(192, 384, 768), out=256, align_corners=False):
        super().__init__()
        self.align = align_corners
        self.lateral_convs = nn.ModuleList([ConvBNReLU(chans[0] + out, out, 1), ConvBNReLU(chans[1] + chans[2], out, 1)])
        self.fpn_convs = nn.ModuleList([ConvBNReLU(out, out, 3), ConvBNReLU(out, out, 3)])

    def forward(self, xs):
        lat = list(xs)
        for i in (2, 1):
            up = F.interpolate(lat[i], size=lat[i - 1].shape[2:], mode="bilinear", align_corners=self.align)
            lat[i - 1] = self.fpn_convs[i - 1](self.lateral_convs[i - 1](torch.cat([lat[i - 1], up], 1)))
        return lat[:2]


class BEVFusionImageSide(nn.Module):
    def __init__(self, n_depth=118, n_ctx=80, align_corners=False):
        super().__init__()
        self.backbone, self.neck = SwinT(), GeneralizedLSSFPN(align_corners=align_corners)
        self.depthnet = nn.Conv2d(256, n_depth + n_ctx, 1)
        self.n_depth = n_depth

    def image_features(self, img):
        f = self.neck(self.backbone(img))[0]
        y = self.depthnet(f)
        return y[:, :self.n_depth].softmax(1), y[:, self.n_depth:]

    forward = image_features

    def load_pretrained(self, path=CKPT):
        sd = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
        take = {}
        for k, v in sd.items():
            for src, dst in (("encoders.camera.backbone.", "backbone."), ("encoders.camera.neck.", "neck."),
                             ("encoders.camera.vtransform.depthnet.", "depthnet.")):
                if k.startswith(src):
                    take[dst + k[len(src):]] = v
        missing, unexpected = [], []
        own = self.state_dict()
        for k in own:
            if k not in take:
                missing.append(k)
        for k, v in take.items():
            if k not in own:
                unexpected.append(k)
            elif own[k].shape != v.shape:
                raise RuntimeError(f"shape mismatch {k}: {tuple(own[k].shape)} vs {tuple(v.shape)}")
        self.load_state_dict({k: v for k, v in take.items() if k in own}, strict=False)
        return {"loaded": len(take) - len(unexpected), "missing": missing, "unexpected": unexpected,
                "frustum_depths": sd["encoders.camera.vtransform.frustum"][:, 0, 0, 2]}


# ----------------------------------------------------------------------------------------------- self-test
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
