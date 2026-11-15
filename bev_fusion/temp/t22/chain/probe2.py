import sys, os
import torch
torch.set_num_threads(2)
sys.path[:0] = ["training", "tests", "tests/support"]
import student_encoder as S
from feature_cache import CachedBEVFusionModel
from model import ModelConfig
from test_slots import BASE, batch_for
D = dict(BASE, camera_features="dual")
torch.manual_seed(0)
real = S.StudentEncoder("resnet18", pretrained=False)
torch.save({"state": {"model." + k: v for k, v in real.state_dict().items()}, "ema": {k: v for k, v in real.state_dict().items() if v.dtype.is_floating_point}, "step": 1, "cfg": {}}, "temp/t22/chain/x.pt")
fro = CachedBEVFusionModel(ModelConfig(**dict(D, camera_student="resnet18", student_init="temp/t22/chain/x.pt")))
b = batch_for(ModelConfig(**D), B=2, Fn=2)
b["lane_img"] = torch.randint(0, 256, (2, 3, 256, 704), dtype=torch.uint8)
fro.train()
out = fro.camera_bevs(b)
print("out requires grad", out.requires_grad, out.abs().sum())
out.sum().backward()
for n, p in [(n, p) for n, p in fro.named_parameters() if n.startswith(("cam_", "lift", "student"))][:12]:
    if p.requires_grad:
        print(n, None if p.grad is None else float(p.grad.abs().sum()))
