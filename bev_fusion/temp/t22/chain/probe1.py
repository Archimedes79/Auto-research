import sys, time, os
import torch
torch.set_num_threads(2)
sys.path[:0] = ["training", "tests"]
from feature_cache import CachedBEVFusionModel
from model import ModelConfig
from test_slots import BASE, batch_for
cfg = ModelConfig(**dict(BASE, camera_features="dual", camera_student="resnet18", student_trainable=True))
torch.manual_seed(0)
t = time.time()
m = CachedBEVFusionModel(cfg).train()
print("built", time.time() - t, "s; params student", sum(p.numel() for p in m.student.parameters()) / 1e6)
b = batch_for(cfg, B=2, Fn=3)
b["lane_img"] = torch.randint(0, 256, (2, 3, 256, 704), dtype=torch.uint8)
t = time.time()
bev = m.camera_bevs(b)
print("camera_bevs", time.time() - t, bev.shape, sorted(m._student_side))
t = time.time()
bev.sum().backward()
print("backward", time.time() - t)
bn = [mod for mod in m.student.modules() if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm)]
print("bn modules", len(bn), "any training:", any(x.training for x in bn))
none = [n for n, p in m.student.named_parameters() if p.grad is None or p.grad.abs().max() == 0]
print("no grad:", none)
