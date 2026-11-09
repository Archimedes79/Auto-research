"""Why does the shuffled-depth control of tests/test_s1_pipeline.py keep most of the gain (vehicle IoU 0.477 against 0.286 for P)? The shuffled depth is the
NEXT sample's histogram: in the synthetic world every scene has the same camera over the same flat road, so the ground depth of a pixel group is the same in
every sample. Measured on the 60 val samples: the share of non-sky 16 x 16 pixel groups whose expected depth in the shuffled variant is within 1 m (and 3 m)
of the true one, split by the group's content (vehicle fraction > 0.5 or not).
Run: wsl ~/miniconda3/envs/gcar/bin/python temp/checks/check_shuffle_control.py"""
import sys

import torch

sys.path.insert(0, "tests")
sys.path.insert(0, "training")
import synthetic_s1_data as D  # noqa: E402
from camera_encoder import depth_centres  # noqa: E402

torch.set_grad_enabled(False)
data = D.load()
samples = data["val"]
prob = torch.stack([s["prob"] for s in samples]).float()
ctx = torch.stack([s["ctx"] for s in samples]).float()
shuf = D.depth_variant(prob, "shuffled")
dc = depth_centres()
ed = lambda p: (p * dc.view(1, -1, 1, 1)).sum(1)                             # noqa: E731
e_true, e_shuf = ed(prob), ed(shuf)
nonsky = prob.sum(1) > 0.5
veh = ctx[:, 3] > 0.5                                                         # context channel 3 = vehicle fraction
d = (e_true - e_shuf).abs()
for name, m in (("all non-sky groups", nonsky), ("groups that are mostly vehicle", nonsky & veh), ("ground groups (no vehicle, no pedestrian)", nonsky & (ctx[:, 3] < 0.01) & (ctx[:, 4] < 0.01))):
    print(f"{name:44s} n = {int(m.sum()):6d}: |expected depth shuffled - true| < 1 m in {float((d[m] < 1).float().mean()):.2f}, < 3 m in {float((d[m] < 3).float().mean()):.2f}, median {float(d[m].median()):.2f} m")
