"""The pre-trained DualBEV single-frame R50 detector (dualbev-r50-cbgs, EMA weights) in plain PyTorch, run with ONLY the
front camera: the state-of-the-art camera reference for the front area (experiment t4_soa_front).

Chain (DualBEV main setting, source read in bev_fusion/literature/code/dualbev.md, sections 2.1-2.3):
  image 256 x 704 -> ResNet-50 -> CustomFPN (512, 16 x 44) -> DepthNet (118 depth logits, 2 semantic logits, 80 context)
  -> depth softmax, bins below 1/D zeroed; context masked by the foreground probability >= 0.25
  -> two BEV pools onto 128 x 128 cells of 0.8 m (+-51.2 m): LSS (frustum, one cell per frustum point) and HeightTrans
     (13 fixed heights projected into the image, nearest depth bin / pixel); the pools sum over heights and cameras
  -> DualFeatFusion (MS_CAM weights w: w * LSS + (1 - w) * HT) -> + learned positional encoding -> ProbNet (BEV logit)
  -> geometry attention sigmoid(conv(mean, max) + logit) * feature -> BEV encoder (ResNet pyramid + FPN_LSS, 256 ch)
  -> CenterHead (shared 3x3 conv 64, six task heads: car | truck, construction vehicle | bus, trailer | barrier |
     motorcycle, bicycle | pedestrian, traffic cone).
The module and tensor names equal the checkpoint's: every tensor loads (only the loss buffer prob.ce_loss.pos_weight is left over).

Differences to the original, stated: (1) the other five cameras are absent (the front area is what is visible), (2) test
boxes come from 3 x 3 peak extraction + a score threshold, not from the rotated-IoU NMS of the original test config,
(3) heading and size are decoded as in CenterPoint but not used by the front-area comparison (centre-distance AP only).
Run the self-test: python training/legacy/dualbev_front.py   (needs models/pretrained/dualbev-r50-cbgs.pth)
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

_TRAINING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # moved to training/legacy/ (2026-10-31): the live blocks
if _TRAINING not in sys.path:
    sys.path.insert(0, _TRAINING)
from bev_trunk import BEVBackbone, FPNLSS  # noqa: E402
from camera_encoder import CustomFPN, DepthNet  # noqa: E402

N_DEPTH, D0, DSTEP = 118, 1.0, 0.5
BEV_N, BEV_RES, BEV_MIN = 128, 0.8, -51.2
Z_MIN, Z_RANGE = -5.0, 8.0
IMG_H, IMG_W, FEAT_H, FEAT_W = 256, 704, 16, 44
TASKS = (("car",), ("truck", "construction_vehicle"), ("bus", "trailer"), ("barrier",), ("motorcycle", "bicycle"),
         ("pedestrian", "traffic_cone"))
GROUP_OF = {"car": "vehicle", "truck": "vehicle", "construction_vehicle": "vehicle", "bus": "vehicle",
            "trailer": "vehicle", "motorcycle": "two_wheeler", "bicycle": "two_wheeler", "pedestrian": "pedestrian"}


# ------------------------------------------------------------------------------------------------ view transformer parts
class MSCAM(nn.Module):
    def __init__(self, cin, cout, r=4):
        super().__init__()
        mid = cin // r
        self.local_att = nn.Sequential(nn.Conv2d(cin, mid, 1), nn.BatchNorm2d(mid), nn.ReLU(True),
                                       nn.Conv2d(mid, cout, 1), nn.BatchNorm2d(cout))
        self.global_att = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(cin, mid, 1), nn.BatchNorm2d(mid),
                                        nn.ReLU(True), nn.Conv2d(mid, cout, 1), nn.BatchNorm2d(cout))

    def forward(self, x):
        return torch.sigmoid(self.local_att(x) + self.global_att(x))


class DualFeatFusion(nn.Module):
    def __init__(self, c=80):
        super().__init__()
        self.ca = MSCAM(2 * c, c)

    def forward(self, x1, x2):
        w = self.ca(torch.cat([x1, x2], 1))
        return w * x1 + (1 - w) * x2


class ChannelAtt(nn.Module):
    def __init__(self, c, r=20):
        super().__init__()
        self.fc = nn.Sequential(nn.Conv2d(c, max(c // r, 2), 1, bias=False), nn.ReLU(True),
                                nn.Conv2d(max(c // r, 2), c, 1, bias=False))

    def forward(self, x):
        return torch.sigmoid(self.fc(F.adaptive_avg_pool2d(x, 1)) + self.fc(F.adaptive_max_pool2d(x, 1)))


class SpatialAtt(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        return torch.sigmoid(self.conv1(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))


class ResCBAMBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c)
        self.conv2, self.bn2 = nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c)
        self.ca, self.sa = ChannelAtt(c), SpatialAtt()

    def forward(self, x):
        y = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        y = y * self.ca(y)
        y = y * self.sa(y)
        return F.relu(y + x)


class ProbNet(nn.Module):
    def __init__(self, cin=80, c=40):
        super().__init__()
        self.base_conv = nn.Sequential(nn.Conv2d(cin, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(True))
        self.prob_conv = nn.Sequential(ResCBAMBlock(c))
        self.mask_net = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        return self.mask_net(self.prob_conv(self.base_conv(x)))


class BEVGeomAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x, logit):
        return torch.sigmoid(self.conv1(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)) + logit)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, num_feats=40, n=BEV_N):
        super().__init__()
        self.row_embed, self.col_embed = nn.Embedding(n, num_feats), nn.Embedding(n, num_feats)

    def forward(self, h, w, device):
        x, y = torch.arange(w, device=device), torch.arange(h, device=device)
        pos = torch.cat([self.col_embed(x)[None].repeat(h, 1, 1), self.row_embed(y)[:, None].repeat(1, w, 1)], -1)
        return pos.permute(2, 0, 1)[None]


def _heights():
    zl, zg = torch.linspace(3, Z_RANGE - 1, 5), torch.linspace(0.5, Z_RANGE - 0.5, 8)
    return torch.cat([zl, zg]) / Z_RANGE * Z_RANGE + Z_MIN              # 13 heights in metres


class ViewTransformer(nn.Module):
    """DualViewTransformerFull for one camera: forward(neck feature, camera vector, K, T_cam_car, ida)."""

    def __init__(self):
        super().__init__()
        self.depth_net = DepthNet()
        self.fuser, self.prob, self.geom_att = DualFeatFusion(80), ProbNet(80, 40), BEVGeomAttention()
        self.positional_encoding = LearnedPositionalEncoding(40)
        d, v, u = torch.meshgrid(D0 + DSTEP * torch.arange(N_DEPTH, dtype=torch.float32),
                                 torch.linspace(0, IMG_H - 1, FEAT_H), torch.linspace(0, IMG_W - 1, FEAT_W), indexing="ij")
        self.register_buffer("frustum", torch.stack([u, v, d], -1), persistent=False)          # (D, H, W, 3), input pixels
        xs = (torch.arange(BEV_N, dtype=torch.float32) + 0.5) / BEV_N * (BEV_N * BEV_RES) + BEV_MIN
        yy, xx = torch.meshgrid(xs, xs, indexing="ij")                                           # row = y, column = x
        pts = torch.stack([xx.reshape(-1), yy.reshape(-1)], -1)                                  # (HW, 2)
        h = _heights()
        ref = torch.cat([pts[None].expand(len(h), -1, -1), h[:, None, None].expand(-1, pts.shape[0], 1)], -1)
        self.register_buffer("ref_points", ref.reshape(-1, 3), persistent=False)                 # (13 HW, 3) ego metres
        self.register_buffer("ref_cell", torch.arange(BEV_N * BEV_N).repeat(len(h)), persistent=False)

    # --- one sample: the two pools onto (HW, C)
    def _lss(self, depth, feat, K, T_cam_car, ida):
        T_c2e = torch.linalg.inv(T_cam_car)
        uv = self.frustum[..., :2] - ida[:2, 2]
        uv = uv @ torch.linalg.inv(ida[:2, :2]).T                                                # undo the image transform
        pc = torch.cat([uv, torch.ones_like(uv[..., :1])], -1) * self.frustum[..., 2:3]
        p = (pc @ torch.linalg.inv(K).T) @ T_c2e[:3, :3].T + T_c2e[:3, 3]
        ix = torch.floor((p[..., 0] - BEV_MIN) / BEV_RES).long()
        iy = torch.floor((p[..., 1] - BEV_MIN) / BEV_RES).long()
        iz = torch.floor((p[..., 2] - Z_MIN) / Z_RANGE).long()
        ok = (ix >= 0) & (ix < BEV_N) & (iy >= 0) & (iy < BEV_N) & (iz == 0)
        vals = depth[..., None] * feat.permute(1, 2, 0)[None]                                    # (D, H, W, C)
        out = torch.zeros(BEV_N * BEV_N, feat.shape[0], device=feat.device, dtype=feat.dtype)
        return out.index_add_(0, (iy * BEV_N + ix)[ok], vals[ok])

    def _ht(self, depth, feat, K, T_cam_car, ida):
        p = self.ref_points @ T_cam_car[:3, :3].T + T_cam_car[:3, 3]
        z = p[:, 2]
        pix = p @ K.T
        uv = pix[:, :2] / torch.clamp(pix[:, 2:3], min=1e-5)
        uv = uv @ ida[:2, :2].T + ida[:2, 2]
        un, vn, zn = uv[:, 0] / IMG_W, uv[:, 1] / IMG_H, (z - D0) / (N_DEPTH * DSTEP)
        ok = (z > 1e-5) & (un > 0) & (un < 1) & (vn > 0) & (vn < 1) & (zn > 0) & (zn < 1)
        ui = (un * FEAT_W).round().long().clamp(0, FEAT_W - 1)
        vi = (vn * FEAT_H).round().long().clamp(0, FEAT_H - 1)
        di = (zn * N_DEPTH).round().long().clamp(0, N_DEPTH - 1)
        w = depth[di, vi, ui]
        vals = w[:, None] * feat[:, vi, ui].T
        out = torch.zeros(BEV_N * BEV_N, feat.shape[0], device=feat.device, dtype=feat.dtype)
        return out.index_add_(0, self.ref_cell[ok], vals[ok])

    def forward(self, x, cam_vec, K, T_cam_car, ida):
        d, ctx = self.depth_net(x, cam_vec)
        depth, sem = d[:, :N_DEPTH].softmax(1), d[:, N_DEPTH:].softmax(1)
        depth = torch.where(depth < 1.0 / N_DEPTH, torch.zeros_like(depth), depth)
        ctx = ctx * (sem[:, 1:2] >= 0.25)
        lss, ht = [], []
        for b in range(x.shape[0]):
            lss.append(self._lss(depth[b], ctx[b], K[b], T_cam_car[b], ida).T.reshape(-1, BEV_N, BEV_N))
            ht.append(self._ht(depth[b], ctx[b], K[b], T_cam_car[b], ida).T.reshape(-1, BEV_N, BEV_N))
        feat = self.fuser(torch.stack(lss), torch.stack(ht))
        logit = self.prob(feat + self.positional_encoding(BEV_N, BEV_N, x.device))
        return self.geom_att(feat, logit) * feat, logit


# ------------------------------------------------------------------------------------------------ head
class ConvBN(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv, self.bn = nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class SeparateHead(nn.Module):
    def __init__(self, c, n_cls):
        super().__init__()
        for name, n in (("reg", 2), ("height", 1), ("dim", 3), ("rot", 2), ("vel", 2), ("heatmap", n_cls)):
            setattr(self, name, nn.Sequential(ConvBN(c, c), nn.Conv2d(c, n, 3, padding=1)))

    def forward(self, x):
        return {k: getattr(self, k)(x) for k in ("reg", "height", "dim", "rot", "vel", "heatmap")}


class CenterHead(nn.Module):
    def __init__(self, cin=256, c=64):
        super().__init__()
        self.shared_conv = ConvBN(cin, c)
        self.task_heads = nn.ModuleList([SeparateHead(c, len(t)) for t in TASKS])

    def forward(self, x):
        s = self.shared_conv(x)
        return [h(s) for h in self.task_heads]


class DualBEVFront(nn.Module):
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet50(weights=None)
        self.img_backbone = nn.Module()
        for name in ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"):
            setattr(self.img_backbone, name, getattr(r, name))
        self.img_neck = CustomFPN()
        self.img_view_transformer = ViewTransformer()
        self.img_bev_encoder_backbone = BEVBackbone(80)
        self.img_bev_encoder_neck = FPNLSS(self.img_bev_encoder_backbone.chans, original_order=True)
        self.pts_bbox_head = CenterHead()

    def forward(self, img, cam_vec, K, T_cam_car, ida):
        b = self.img_backbone
        c3 = b.layer3(b.layer2(b.layer1(b.maxpool(b.relu(b.bn1(b.conv1(img)))))))
        c4 = b.layer4(c3)
        bev, _ = self.img_view_transformer(self.img_neck(c3, c4), cam_vec, K, T_cam_car, ida)
        return self.pts_bbox_head(self.img_bev_encoder_neck(self.img_bev_encoder_backbone(bev)))

    def load_pretrained(self, path):
        sd = torch.load(path, map_location="cpu", weights_only=True)
        sd = sd.get("state_dict", sd)
        res = self.load_state_dict(sd, strict=False)
        return {"missing": [k for k in res.missing_keys], "unexpected": list(res.unexpected_keys)}


@torch.no_grad()
def decode(outs, k=100, threshold=0.05):
    """list (per sample) of box dicts (group, score, cx, cy, vx, vy, length, width, yaw) for the groups of GROUP_OF:
    3 x 3 peaks of the sigmoid heatmaps, the best k over all classes, score >= threshold."""
    B = outs[0]["heatmap"].shape[0]
    res = [[] for _ in range(B)]
    for t, names in enumerate(TASKS):
        o = outs[t]
        hm = torch.sigmoid(o["heatmap"])
        hm = hm * (hm == F.max_pool2d(hm, 3, 1, 1))
        for ci, name in enumerate(names):
            if name not in GROUP_OF:
                continue
            sc, idx = hm[:, ci].reshape(B, -1).topk(k)
            for b in range(B):
                for s, i in zip(sc[b].tolist(), idx[b].tolist()):
                    if s < threshold:
                        break
                    iy, ix = divmod(i, BEV_N)
                    ox, oy = o["reg"][b, :, iy, ix].tolist()
                    dm = o["dim"][b, :, iy, ix].exp().tolist()
                    sn, cs = o["rot"][b, :, iy, ix].tolist()
                    res[b].append({"group": GROUP_OF[name], "cls": name, "score": s,
                                   "cx": (ix + ox) * BEV_RES + BEV_MIN, "cy": (iy + oy) * BEV_RES + BEV_MIN,
                                   "vx": o["vel"][b, 0, iy, ix].item(), "vy": o["vel"][b, 1, iy, ix].item(),
                                   "length": dm[0], "width": dm[1], "yaw": math.atan2(sn, cs)})
    for b in range(B):
        res[b] = sorted(res[b], key=lambda d: -d["score"])[:k]
    return res


def _selftest():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "pretrained", "dualbev-r50-cbgs.pth")
    m = DualBEVFront().eval()
    rep = m.load_pretrained(path)
    assert rep["missing"] == [] and rep["unexpected"] == ["img_view_transformer.prob.ce_loss.pos_weight"], rep
    # geometry: a point 20 m ahead of an ideal front camera lands in the middle BEV column of the pools
    K = torch.tensor([[1266.4, 0, 800.0], [0, 1266.4, 450.0], [0, 0, 1.0]])
    R = torch.tensor([[0.0, 0, 1], [-1, 0, 0], [0, -1, 0]])            # camera x right, y down, z forward -> ego x forward
    T_c2e = torch.eye(4)
    T_c2e[:3, :3], T_c2e[:3, 3] = R, torch.tensor([1.5, 0.0, 1.5])
    s = 704 / 1600
    ida = torch.tensor([[s, 0, 0], [0, s, -(900 * s - 256)], [0, 0, 1.0]])
    vt = m.img_view_transformer
    depth = torch.zeros(N_DEPTH, FEAT_H, FEAT_W)
    feat = torch.ones(80, FEAT_H, FEAT_W)
    depth[int((21.5 - D0) / DSTEP), :, :] = 1.0                        # all mass at 21.5 m depth (21.5 m along z; 20 m ahead of the rig)
    for name, fn in (("lss", vt._lss), ("ht", vt._ht)):
        out = fn(depth, feat, K, torch.linalg.inv(T_c2e), ida).sum(1).reshape(BEV_N, BEV_N)
        yy, xx = torch.nonzero(out, as_tuple=True)
        assert len(xx) > 0, name
        xm, ym = (xx.float().mean() + 0.5) * BEV_RES + BEV_MIN, (yy.float().mean() + 0.5) * BEV_RES + BEV_MIN
        assert abs(xm - 23.0) < 2.5 and abs(ym) < 1.5, (name, xm.item(), ym.item())
    img, cv = torch.randn(1, 3, IMG_H, IMG_W), torch.randn(1, 27)
    with torch.no_grad():
        outs = m(img, cv, K[None], torch.linalg.inv(T_c2e)[None], ida)
    assert [tuple(o["heatmap"].shape) for o in outs] == [(1, len(t), BEV_N, BEV_N) for t in TASKS]
    dets = decode(outs, k=20, threshold=0.0)[0]
    assert len(dets) == 20 and {"cx", "cy", "score", "group"} <= set(dets[0])
    print("dualbev_front self-test passed:", rep)


if __name__ == "__main__":
    _selftest()
