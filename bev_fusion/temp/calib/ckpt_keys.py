"""One-off (front-area SOA comparison): tensor names and shapes of the DualBEV checkpoint parts we do not have code for."""
import os
import sys
import torch
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "pretrained", "dualbev-r50-cbgs.pth")
sd = torch.load(p, map_location="cpu", weights_only=True)
sd = sd.get("state_dict", sd)
for pre in sys.argv[1:]:
    for k, v in sd.items():
        if k.startswith(pre):
            print(k, tuple(v.shape))
