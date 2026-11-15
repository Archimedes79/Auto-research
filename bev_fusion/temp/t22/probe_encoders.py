"""Scratch (CPU, seconds): feature levels (stride, channels) and parameter counts of the two image-encoder candidates at 256 x 704,
and a check that the downloaded EfficientViT-B1 weights load into timm's features_only model.
Run from bev_fusion/: wsl /home/ml/miniconda3/envs/gcar/bin/python temp/t22/probe_encoders.py"""
import torch
import timm
from safetensors.torch import load_file

torch.set_num_threads(2)
x = torch.zeros(1, 3, 256, 704)

m = timm.create_model("efficientvit_b1.r256_in1k", pretrained=False, features_only=True)
sd = load_file("models/pretrained/imagenet/efficientvit_b1_r256_in1k/model.safetensors")
missing, unexpected = m.load_state_dict(sd, strict=False)
print("efficientvit_b1: missing", len(missing), missing[:4], "unexpected", len(unexpected), unexpected[:4])
print("  feature_info:", [(i["reduction"], i["num_chs"]) for i in m.feature_info])
with torch.no_grad():
    outs = m(x)
print("  output shapes at 256x704:", [tuple(o.shape[1:]) for o in outs])
print("  params (M):", round(sum(p.numel() for p in m.parameters()) / 1e6, 2))

import torchvision
r = torchvision.models.resnet18()
r.load_state_dict(torch.load("models/pretrained/imagenet/resnet18-f37072fd.pth", weights_only=True))
feats = {}
y = r.maxpool(r.relu(r.bn1(r.conv1(x))))
chs = []
for name in ("layer1", "layer2", "layer3", "layer4"):
    y = getattr(r, name)(y)
    chs.append((x.shape[-1] // y.shape[-1], y.shape[1]))
print("resnet18: (stride, channels) of layer1..4:", chs, " params (M):", round(sum(p.numel() for p in r.parameters()) / 1e6, 2))
