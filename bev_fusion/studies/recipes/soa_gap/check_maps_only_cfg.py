"""Gate for the maps-only arm (known answers, CPU, no data): with the pinned code of the reference run (commit 0a3d73c, extracted to argv[1]) the
config t6_maps_only builds, the task weights are fixed (1, 1, 0, 0), the radar aux head has weight 0, and the gradient of the objects and
motion groups with respect to a shared parameter is exactly zero while the areas / lines groups give a non-zero one.
Usage: python temp/soa_gap/check_maps_only_cfg.py <dir with the extracted training/ of the pinned commit> training/configs/t6_maps_only.json"""
import json
import sys

import torch

sys.path.insert(0, sys.argv[1])
import train as T  # noqa: E402

raw = json.load(open(sys.argv[2]))
cfg = T.TrainConfig(**raw)
model, loss, _ = T.build(cfg)
assert loss.task.mode == "fixed" and tuple(loss.task.fixed) == (1.0, 1.0, 0.0, 0.0), (loss.task.mode, loss.task.fixed)
assert loss.cfg.w_radar_aux == 0.0 and loss.cfg.w_cam_aux == 0.3
print("config builds; task weights", loss.task.fixed, "; radar aux", loss.cfg.w_radar_aux, "; camera aux", loss.cfg.w_cam_aux)
w = torch.nn.Parameter(torch.ones(()))
groups = {g: w * (i + 1.0) for i, g in enumerate(loss.task.groups)}
total, used = loss.task(groups)
(g,) = torch.autograd.grad(total, w)
assert abs(float(g) - (1.0 * 1 + 1.0 * 2)) < 1e-6, float(g)      # d/dw (1*w + 1*2w + 0*3w + 0*4w) = 3
for i, name in enumerate(loss.task.groups):
    (gi,) = torch.autograd.grad(loss.task({name: w * 1.0})[0], w)
    print("  d total / d w from group %-8s = %.1f" % (name, float(gi)))
print("gate ok: objects and motion groups carry weight 0 (no gradient), areas and lines weight 1")
