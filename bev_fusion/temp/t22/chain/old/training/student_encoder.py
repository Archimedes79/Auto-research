"""Student image encoder (experiment t22, stage A): ONE small ImageNet-initialised encoder that outputs the two interfaces of the cached frozen teachers, so the chain
downstream (lifts, trunk, heads) and its warm start stay valid (card: experiments/t22_student_encoder/experiment.md).

    image (B, 3, 256, 704) RGB 0..255, uint8 or float (the DualBEV crop that `data/cache/img_front` holds), normalised with mean (123.675, 116.28, 103.53) /
    std (58.395, 57.12, 57.375) (the DualBEV convention of camera_encoder.py)
      -> backbone: timm EfficientViT-B1 (`features_only`: strides 4 / 8 / 16 / 32, 32 / 64 / 128 / 256 ch, 4.6 M parameters, weights from the local file) or, for the control
         arm, torchvision ResNet-18 (64 / 128 / 256 / 512 ch); nothing is downloaded
      -> light top-down FPN, 128 ch, from the strides 8 / 16 / 32: 1 x 1 lateral + nearest upsample-add, then a 3 x 3 conv at strides 16 and 8
      -> H16 on the stride-16 map, H8 on the stride-8 map; each = 3 x 3 conv (BN, ReLU), then 1 x 1 to 118 depth logits + 80 context channels

The heads output the SAME tensors the caches hold, in the conventions of the two teachers (they differ, and so does the image window):
  H16 = DualBEV (camera_encoder.py, cache data/cache/cam_feats): 16 x 44 cells of the JPEG's own window (`nuscenes_front_ida`: resize 0.44, crop the top 140 rows), cell k at
        pixel 16 k + 7.5 (cell centre); 118 depth bins 1.0 .. 60 m in 0.5 m steps, softmax = probability at the bin centres 1.25 + 0.5 i (`depth_centres`); 80 context channels.
  H8  = BEVFusion (bevfusion_camera.py, cache data/cache/cam_feats_bevfusion): 32 x 88 cells in FRUSTUM convention (cell j at pixel j (W - 1) / (w - 1)), depth bins at the
        LEFT edges 1.0 + 0.5 i (`DEPTHS`), 80 context channels, AND in BEVFusion's own window (`bevfusion_ida`: resize 0.48, crop x0 32 / y0 176), which shows a smaller, 9 % magnified
        part of the original image than the JPEG. The student reads the JPEG only, so its stride-8 map is computed in the JPEG's window (cell centres 8 k + 3.5) and `h8_warp_grid`
        resamples it, bilinearly and without parameters, onto the cells of BEVFusion's frustum: a cell then carries the prediction for the image point the teacher's cell looks at (a
        position-dependent shift of up to +-4 cells that a convolution cannot learn). `warp_h8=False` leaves H8 in the JPEG's window (stage B may lift it with the DualBEV ida).
No camera-parameter gate in this first version (DualBEV's DepthNet gates its depth and context branches with the 27-vector of the camera; the front camera is one rig).
Cells are assumed at the centre convention above; a constant sub-cell offset of the net's own alignment is absorbed by the learned 3 x 3 convs.

forward(img) -> {"prob16" (B, 118, 16, 44), "ctx16" (B, 80, 16, 44), "prob8" (B, 118, 32, 88), "ctx8" (B, 80, 32, 88), "logit16", "logit8"} (probabilities = softmax of the logits
over the 118 bins). Test: tests/test_student_encoder.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

try:
    from .bevfusion_camera import bevfusion_ida, frustum_pixels
    from .camera_encoder import IMG_MEAN, IMG_STD, nuscenes_front_ida
except ImportError:                                   # run as a script
    from bevfusion_camera import bevfusion_ida, frustum_pixels
    from camera_encoder import IMG_MEAN, IMG_STD, nuscenes_front_ida

N_DEPTH, N_CTX, IMG_HW = 118, 80, (256, 704)
HW8 = (32, 88)                                         # the cache size of H8 (feature_cache.FEAT_HW_OF); H16 is 16 x 44
WEIGHTS = {"efficientvit_b1": "models/pretrained/imagenet/efficientvit_b1_r256_in1k/model.safetensors",
           "resnet18": "models/pretrained/imagenet/resnet18-f37072fd.pth"}


# ------------------------------------------------------------------------------------------------ backbones
class ResNetFeatures(nn.Module):
    """torchvision ResNet-18 as a feature pyramid: the outputs of layer1 .. layer4 (strides 4 / 8 / 16 / 32)."""

    def __init__(self, pretrained=True):
        super().__init__()
        r = torchvision.models.resnet18(weights=None)
        if pretrained:                                 # third-party file: tensors only
            r.load_state_dict(torch.load(WEIGHTS["resnet18"], map_location="cpu", weights_only=True))
        self.conv1, self.bn1, self.relu, self.maxpool = r.conv1, r.bn1, r.relu, r.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4

    def forward(self, x):
        c1 = self.layer1(self.maxpool(self.relu(self.bn1(self.conv1(x)))))
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        return [c1, c2, c3, self.layer4(c3)]


def make_backbone(name, pretrained=True):
    """(module returning the four feature maps of strides 4 / 8 / 16 / 32, their channels)"""
    if name == "efficientvit_b1":
        import timm
        kw = dict(pretrained_cfg_overlay=dict(file=WEIGHTS[name])) if pretrained else {}     # a local file: no hub access
        net = timm.create_model("efficientvit_b1.r256_in1k", pretrained=pretrained, features_only=True, **kw)
        return net, tuple(net.feature_info.channels())
    if name == "resnet18":
        return ResNetFeatures(pretrained), (64, 128, 256, 512)
    raise ValueError(f"backbone {name!r}: one of {sorted(WEIGHTS)}")


# ------------------------------------------------------------------------------------------------ geometry of H8
def h8_warp_grid(stride=8, out_hw=HW8, hw=IMG_HW):
    """(grid (1, h, w, 2) for F.grid_sample with align_corners=True, xy (h, w, 2)): output cell (i, j) of BEVFusion's frustum -> its pixel (x, y) in the JPEG's (DualBEV) window
    -> its position in the native stride-`stride` map (cell centres at stride * k + (stride - 1) / 2), normalised to [-1, 1]. The pixel comes from the two ida matrices
    (original image -> window), exactly as the lifts use them: p_dualbev = ida_d @ inv(ida_b) @ p_bevfusion."""
    fh, fw = out_hw
    xs, ys = frustum_pixels(fh, fw, *hw)               # BEVFusion-window pixels of the frustum cells
    pts = torch.stack([xs.view(1, fw).expand(fh, fw), ys.view(fh, 1).expand(fh, fw), torch.ones(fh, fw)], -1)
    xy = (pts @ (nuscenes_front_ida(dst_hw=hw) @ torch.linalg.inv(bevfusion_ida())).T)[..., :2]
    cell = (xy - (stride - 1) / 2) / stride            # native cell index (fractional)
    size = torch.tensor([hw[1] // stride, hw[0] // stride], dtype=torch.float32)       # (w, h) of the native map
    return (2 * cell / (size - 1) - 1).unsqueeze(0), xy


# ------------------------------------------------------------------------------------------------ the encoder
class Head(nn.Module):
    """3 x 3 conv (128, BN, ReLU), then 1 x 1 to 118 depth logits + 80 context channels (the layout of the caches: probabilities first)."""

    def __init__(self, c):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(True))
        self.out = nn.Conv2d(c, N_DEPTH + N_CTX, 1)

    def forward(self, x):
        return self.out(self.conv(x))


class StudentEncoder(nn.Module):
    def __init__(self, backbone="efficientvit_b1", pretrained=True, c_fpn=128, warp_h8=True):
        super().__init__()
        self.backbone, chans = make_backbone(backbone, pretrained)
        self.lateral = nn.ModuleList([nn.Conv2d(c, c_fpn, 1) for c in chans[1:]])                 # strides 8, 16, 32
        self.smooth = nn.ModuleList([nn.Conv2d(c_fpn, c_fpn, 3, padding=1) for _ in range(2)])    # strides 8, 16
        self.head8, self.head16 = Head(c_fpn), Head(c_fpn)
        self.register_buffer("mean", torch.tensor(IMG_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMG_STD).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("grid8", h8_warp_grid()[0] if warp_h8 else None, persistent=False)
        self._frozen = False

    def encode(self, img):
        """img (B, 3, 256, 704) RGB 0..255 (uint8 or float) -> the four backbone maps (strides 4 / 8 / 16 / 32)"""
        if tuple(img.shape[-2:]) != IMG_HW:
            raise ValueError(f"the conventions of the heads are those of the {IMG_HW} window, got {tuple(img.shape[-2:])}")
        return self.backbone((img.float() - self.mean) / self.std)

    def decode(self, feats):
        """the four backbone maps -> the output dict (FPN, both heads, the H8 resampling, the softmax)"""
        _, c8, c16, c32 = feats
        p32 = self.lateral[2](c32)
        p16 = self.lateral[1](c16) + F.interpolate(p32, size=c16.shape[-2:], mode="nearest")
        p8 = self.lateral[0](c8) + F.interpolate(p16, size=c8.shape[-2:], mode="nearest")
        o16 = self.head16(self.smooth[1](p16))
        o8 = self.head8(self.smooth[0](p8))
        if self.grid8 is not None:                     # border: the last frustum row lies 0.4 cell below the native map
            o8 = F.grid_sample(o8, self.grid8.expand(len(o8), -1, -1, -1), mode="bilinear", padding_mode="border", align_corners=True)
        out = {}
        for k, o in (("16", o16), ("8", o8)):
            out["logit" + k], out["ctx" + k] = o[:, :N_DEPTH], o[:, N_DEPTH:]
            out["prob" + k] = out["logit" + k].softmax(1)
        return out

    def forward(self, img):
        return self.decode(self.encode(img))

    # ---- training
    def freeze_backbone(self, flag=True):
        """flag: the backbone gets no gradient and its BatchNorm stays in eval mode (also under .train()); FPN and heads train."""
        self._frozen = flag
        for p in self.backbone.parameters():
            p.requires_grad_(not flag)
        return self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self._frozen:
            self.backbone.eval()
        return self

    def lr_groups(self, lr, backbone_mult=0.1, prefix=""):
        """trainkit's `Config.lr_groups` for this encoder (`prefix` = its name inside a bigger model): the backbone at backbone_mult x lr, FPN and heads at lr; {} while frozen."""
        return {} if self._frozen else {prefix + "backbone": {"lr": lr * backbone_mult}}
