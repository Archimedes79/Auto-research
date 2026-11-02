"""FLOPs of the blocks of the front-camera chain at the planned sizes (hardware independent; conv / matmul only).

camera encoder (ResNet-50 + neck + DepthNet at 256 x 704, PD lift incl. grid_sample not counted), lift output conv,
splat head and radar block as far as they exist as modules. Run from bev_fusion:
wsl ~/miniconda3/envs/gcar/bin/python temp/checks/chain_flops.py
"""
import sys

import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, "training")
from camera_encoder import FrontCameraEncoder, camera_vector, nuscenes_front_ida  # noqa: E402
from splat import CorrectionHead, VelocityHead  # noqa: E402


def gf(fn):
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        fn()
    return fc.get_total_flops() / 1e9


enc = FrontCameraEncoder().eval()
img = torch.randn(1, 3, 256, 704)
K = torch.tensor([[1266.0, 0, 799.5], [0, 1266.0, 449.5], [0, 0, 1]])
R = torch.tensor([[0.0, -1, 0], [0, 0, -1], [1, 0, 0]])
T = torch.eye(4)
T[:3, :3], T[:3, 3] = R, -R @ torch.tensor([0.0, 0, 1.5])
ida = nuscenes_front_ida()
cam = camera_vector(K[None], ida[None, :2, :2], ida[None, :2, 2], torch.linalg.inv(T)[None])
rows = [("camera: ResNet-50 backbone (256 x 704)", gf(lambda: [enc.img_backbone.layer4(enc.img_backbone.layer3(enc.img_backbone.layer2(enc.img_backbone.layer1(enc.img_backbone.maxpool(enc.img_backbone.relu(enc.img_backbone.bn1(enc.img_backbone.conv1(img)))))))) ]))]
b = enc.img_backbone
with torch.no_grad():
    c3 = b.layer3(b.layer2(b.layer1(b.maxpool(b.relu(b.bn1(b.conv1(img)))))))
    c4 = b.layer4(c3)
    neck = enc.img_neck(c3, c4)
rows.append(("camera: neck (CustomFPN)", gf(lambda: enc.img_neck(c3, c4))))
rows.append(("camera: DepthNet (gate, 3 blocks, ASPP)", gf(lambda: enc.depth_net(neck, cam))))
with torch.no_grad():
    prob, ctx = enc.image_features(img, cam)
rows.append(("camera: lift PD + out conv (grid_sample uncounted)", gf(lambda: enc.out_conv(enc.lift(ctx, prob, K, T, (256, 704), ida)))))
vh, ch = VelocityHead(128).eval(), CorrectionHead(128).eval()
bev = torch.randn(1, 128, 120, 104)
rows.append(("temporal: velocity head, per branch and step (128 ch)", gf(lambda: vh(bev))))
rows.append(("temporal: correction head, per branch and step", gf(lambda: ch(bev.reshape(1, 128, -1), torch.zeros(1, 12480, 2), 0.08, 10.0, 0.1))))
tot = sum(r[1] for r in rows[:4])
print(f"{'block':58s} {'GFLOPs':>8s}")
for n, v in rows:
    print(f"{n:58s} {v:8.2f}")
print(f"{'camera encoder total (first four rows)':58s} {tot:8.2f}")
