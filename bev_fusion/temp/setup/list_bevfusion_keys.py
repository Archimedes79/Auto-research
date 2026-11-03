import torch
sd = torch.load("models/pretrained/bevfusion/camera-only-seg.pth", map_location="cpu", weights_only=True)["state_dict"]
pre = "encoders.camera."
for k, v in sd.items():
    if not k.startswith(pre):
        continue
    s = k[len(pre):]
    if s.startswith("backbone.stages.") and not (".blocks.0." in s or ".blocks.1." in s and s.startswith("backbone.stages.0.") or "downsample" in s):
        continue
    print(s, tuple(v.shape), str(v.dtype).replace("torch.", ""))
