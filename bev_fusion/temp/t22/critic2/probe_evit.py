import sys, time
sys.path[:0] = ["training", "tests", "tests/support", ".", "../autoresearch"]
import torch, torch.nn as nn
torch.set_num_threads(2)
import model as M, student_encoder as S
from feature_cache import CachedBEVFusionModel
from test_slots import BASE, batch_for
cfg = M.ModelConfig(**dict(BASE, camera_features="dual", camera_student="evit_b1", student_trainable=True))
torch.manual_seed(0)
m = CachedBEVFusionModel(cfg)
kinds = {}
for mod in m.student.modules():
    if "Norm" in type(mod).__name__ or "norm" in type(mod).__name__.lower():
        kinds[type(mod).__name__] = kinds.get(type(mod).__name__, 0) + 1
print("norm layers in the evit_b1 student:", kinds)
bn = [x for x in m.student.modules() if isinstance(x, nn.modules.batchnorm._BatchNorm)]
before = [(x.running_mean.clone(), x.running_var.clone(), x.num_batches_tracked.clone()) for x in bn]
m.train()
b = batch_for(cfg, B=2, Fn=2)
b["lane_img"] = torch.randint(0, 256, (2, 3, 256, 704), dtype=torch.uint8)
t0 = time.time()
out = m.camera_bevs(b); out.square().mean().backward()
print(f"fwd+bwd {time.time()-t0:.1f}s")
dead = [n for n, p in m.student.named_parameters() if p.grad is None or float(p.grad.abs().max()) == 0]
print("params", sum(1 for _ in m.student.parameters()), "no gradient:", dead[:6], len(dead))
print("BN in eval:", all(not x.training for x in bn), "stats unchanged:", all(torch.equal(a[0], x.running_mean) and torch.equal(a[1], x.running_var) and torch.equal(a[2], x.num_batches_tracked) for a, x in zip(before, bn)))
print("training flag of the non-BN modules (should follow train()):", sum(1 for x in m.student.modules() if x.training and not isinstance(x, nn.modules.batchnorm._BatchNorm)))
print("dropout/drop-path modules:", [type(x).__name__ for x in m.student.modules() if "rop" in type(x).__name__][:5])
