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

Stage B (in the chain, ModelConfig.camera_student): the same module is `model.student`; this file also holds what the chain needs of it: the distillation loss (`distill_terms`, used by
distill_student.py and, as the anchor, by train.forward_loss: `anchor_terms`, `student_anchor_weight`), the BatchNorm switch (`keep_bn_eval`) and the loader of a trained student
(`load_trained`). Test: tests/test_student_chain.py
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
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
    HEAD_OF = {"dualbev": "16", "bevfusion": "8"}      # model.CAMERAS feature family -> the head that stands in for its cached teacher output (H16: DualBEV, H8: BEVFusion)

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
        self._bn_eval = False
        self.checkpoint_encode = False                 # True: the backbone is recomputed in the backward instead of keeping its activations (set by model._build_student for a student that trains)

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
        # t22 stage B: 9 key frames of a micro-batch (batch 3 x clips of 3, the graph kept over the clip) held the backbone activations of every frame at once: peak 16.3 GB on the 12 GB card
        # (smoke run 2026-11-05, 56 s per step from GPU-memory paging). The encoder is then recomputed in the backward and only its four output maps (2.7 MB per frame) stay. Every backward
        # that reaches a frame recomputes it (with seq_bptt a clip of 3 runs the encoder 3 + 1 + 2 + 3 = 9 times, not 6). Only with BatchNorm held in eval mode (keep_bn_eval): a train-mode
        # BatchNorm would count its statistics twice; the backbone has no dropout, so the recomputation is the same function.
        if self.checkpoint_encode and self._bn_eval and self.training and torch.is_grad_enabled() and any(p.requires_grad for p in self.backbone.parameters()):
            feats = torch.utils.checkpoint.checkpoint(self.encode, img, use_reentrant=False)
        else:
            feats = self.encode(img)
        return self.decode(feats)

    # ---- training
    def freeze_backbone(self, flag=True):
        """flag: the backbone gets no gradient and its BatchNorm stays in eval mode (also under .train()); FPN and heads train."""
        self._frozen = flag
        for p in self.backbone.parameters():
            p.requires_grad_(not flag)
        return self.train(self.training)

    def keep_bn_eval(self, flag=True):
        """flag: every BatchNorm of the encoder stays in eval mode (its running statistics are fixed, its scale and shift still train) whatever .train() of the model around it asks:
        stage B fine-tunes on the few key frames of a micro-batch, which would drag the statistics the distillation fitted"""
        self._bn_eval = flag
        return self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self._frozen:
            self.backbone.eval()
        if self._bn_eval:
            for m in self.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self

    def lr_groups(self, lr, backbone_mult=0.1, prefix=""):
        """trainkit's `Config.lr_groups` for this encoder (`prefix` = its name inside a bigger model): the backbone at backbone_mult x lr, FPN and heads at lr; {} while frozen."""
        return {} if self._frozen else {prefix + "backbone": {"lr": lr * backbone_mult}}


# ------------------------------------------------------------------------------------------------ the loss against the teachers (stage A training, stage B anchor)
def cell_maps(out, tgt):
    """the per-cell terms, each (B, h, w): kl<k> = KL(teacher || student) over the 118 depth bins, cos<k> = cosine similarity of the 80 context channels, mse<k> = their mean squared
    error; k = 16 (DualBEV) and 8 (BEVFusion). The teacher probabilities are renormalised (they are stored as float16)."""
    m = {}
    for k in ("16", "8"):
        t = tgt["p" + k].float()
        t = t / t.sum(1, keepdim=True)
        m["kl" + k] = (t * (t.clamp_min(1e-8).log() - out["logit" + k].float().log_softmax(1))).sum(1)
        c, tc = out["ctx" + k].float(), tgt["c" + k].float()
        m["cos" + k] = F.cosine_similarity(c, tc, dim=1)
        m["mse" + k] = (c - tc).pow(2).mean(1)
    return m


def total_of(mean):
    """the loss from the cell means: kl + (1 - cos) + mse on both heads, equal weights"""
    return sum(mean["kl" + k] + 1 - mean["cos" + k] + mean["mse" + k] for k in ("16", "8"))


def distill_terms(out, tgt):
    """the training terms: kl<k>, cosl<k> (= 1 - cosine), mse<k>, total"""
    mean = {n: v.mean() for n, v in cell_maps(out, tgt).items()}
    terms = {n: v for n, v in mean.items() if not n.startswith("cos")}
    terms.update({"cosl" + k: 1 - mean["cos" + k] for k in ("16", "8")})
    terms["total"] = total_of(mean)
    return terms


# ------------------------------------------------------------------------------------------------ in the chain (stage B)
def anchor_terms(side):
    """distill_terms of the chain's `student_side` (model._forward's result entry): the student's logits and context of the current frame against the cached teacher's probabilities and
    context of the same frame -> the stage-A terms; the anchor of train.forward_loss is its ["total"]"""
    out = {n + k: side[c + k] for k in ("16", "8") for n, c in (("logit", "l"), ("ctx", "c"))}
    tgt = {n + k: side["t_" + n + k] for k in ("16", "8") for n in ("p", "c")}
    return distill_terms(out, tgt)


def student_anchor_weight(step_frac, start, floor, until):
    """the weight of the distillation anchor at the fraction `step_frac` of the steps: `start` at the beginning, falling linearly to start * floor at the fraction `until`, constant after"""
    return start * (1.0 - (1.0 - floor) * min(1.0, max(0.0, step_frac) / until))


def newest_checkpoint(path):
    """the checkpoint a path names: a file as it is, or the newest ckpt_*.pt of a run folder; None when there is none"""
    if os.path.isfile(path):
        return path
    names = sorted(f for f in os.listdir(path) if f.startswith("ckpt_") and f.endswith(".pt")) if os.path.isdir(path) else []
    return os.path.join(path, names[-1]) if names else None


def load_trained(model, path, ck=None):
    """`model` takes the weights of a trained student: the EMA tensors (the weights the validation of the run saw) and the integer buffers of the checkpoint `path` names (a file or a run folder:
    its newest one), strictly: a tensor the checkpoint lacks, one it has too many or of another shape is an error. ck = the checkpoint already read. -> the checkpoint dict"""
    try:
        from trainkit.checkpoint import warm_start     # the engine's loader (autoresearch/trainkit): on sys.path once train.py / distill_student.py is imported, a run's snapshot carries a copy
    except ImportError:                                # else the live autoresearch/ of the repository, found by walking up (as train.py does)
        root = os.path.dirname(os.path.abspath(__file__))
        while not os.path.isdir(os.path.join(root, "autoresearch", "trainkit")) and os.path.dirname(root) != root:
            root = os.path.dirname(root)
        sys.path.append(os.path.join(root, "autoresearch"))
        from trainkit.checkpoint import warm_start
    file = newest_checkpoint(path)
    if file is None:
        raise FileNotFoundError(f"{path}: no checkpoint (a ckpt_*.pt file or a run folder with one)")
    ck = ck if ck is not None else torch.load(file, map_location="cpu", weights_only=False)
    fresh = warm_start(file, model, nn.Module(), ck=ck)
    if fresh:
        raise ValueError(f"{file} lacks {len(fresh)} tensor(s) of this student, e.g. {fresh[:3]}")
    return ck
