"""Scratch (CPU, seconds): does the downloaded EfficientViT-B1 file load into timm's features_only model the official way (pretrained_cfg_overlay file=...)?
Also compares one parameter with the full model loaded by key. Run from bev_fusion/ with the gcar interpreter (see probe_encoders.py)."""
import torch
import timm
from safetensors.torch import load_file

torch.set_num_threads(2)
path = "models/pretrained/imagenet/efficientvit_b1_r256_in1k/model.safetensors"
m = timm.create_model("efficientvit_b1.r256_in1k", pretrained=True, features_only=True, pretrained_cfg_overlay=dict(file=path))
full = timm.create_model("efficientvit_b1.r256_in1k", pretrained=False)
miss, unexp = full.load_state_dict(load_file(path), strict=True), None
print("full model strict load ok:", miss)
a = dict(m.named_parameters())["stem_in_conv.conv.weight"]
b = dict(full.named_parameters())["stem.in_conv.conv.weight"]
print("features_only stem weight equals the file's:", bool(torch.equal(a, b)), " (nonzero:", bool(a.abs().sum() > 0), ")")
x = torch.zeros(1, 3, 256, 704)
with torch.no_grad():
    print([tuple(o.shape[1:]) for o in m(x)])
