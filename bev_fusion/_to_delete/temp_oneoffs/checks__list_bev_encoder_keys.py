"""One-off: layer names and shapes of the BEV encoder in the DualBEV checkpoint (img_bev_encoder_backbone / _neck), to
rebuild it exactly in plain PyTorch.   Run from bev_fusion: wsl ~/miniconda3/envs/gcar/bin/python temp/checks/list_bev_encoder_keys.py
"""
import torch

sd = torch.load("models/pretrained/dualbev-r50-cbgs.pth", map_location="cpu", weights_only=True)["state_dict"]
tot = 0
for k, v in sd.items():
    if k.startswith(("img_bev_encoder_backbone.", "img_bev_encoder_neck.")) and "num_batches" not in k:
        tot += v.numel()
        print(f"{k:62s} {tuple(v.shape)}")
print("parameters in the BEV encoder:", tot)
