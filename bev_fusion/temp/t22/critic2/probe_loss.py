"""usage: probe_loss.py <code root> <out.pt>  -- forward_loss (train and validation terms) + run_clip rows of the option-off cached chain, from a code tree"""
import copy, os, sys
root, out = sys.argv[1], sys.argv[2]
sys.path[:0] = [os.path.join(root, "bev_fusion", "training"), os.path.join(root, "bev_fusion", "tests"), os.path.join(root, "bev_fusion", "tests", "support"), os.path.join(root, "bev_fusion"), os.path.join(root, "autoresearch")]
import torch
torch.set_num_threads(2)
import model as M, train as T, synthetic_world as sw
from feature_cache import CachedBEVFusionModel
from losses import BEVLoss, synthetic_gt
import test_slots as TS
print("train from", T.__file__)
torch.manual_seed(0)
cfgm = M.ModelConfig(**dict(TS.BASE, camera_features="dual", radar_kind="delta_slots", c_rad=64, rad_temporal="window_memory", rad_static_gate=True, object_direction=True,
                            object_moving=True, motion_block=True, box_vel_from_field=True, lane_branch="rgb"))
m = CachedBEVFusionModel(cfgm).train()
with torch.no_grad():
    for q in m.parameters():
        if q.requires_grad and q.abs().max() == 0: q.normal_(0, 1e-2)
loss = BEVLoss(grid=m.grid)
b = TS.batch_for(cfgm, B=2, Fn=3)
b["lane_img"] = torch.randint(0, 256, (2, 3, *TS.DST_HW), dtype=torch.uint8, generator=torch.Generator().manual_seed(4))
B_ = sw.Box
scene = sw.Scene([B_(6.0, -1.75, 4.5, 1.8, 1.5, 0.0, 8.0, 0.0, "car", 10.0), B_(10.0, 4.0, 4.5, 1.8, 1.5, 0.0, 0.0, 0.0, "car", 10.0)], lanes=[-5.25, -1.75, 1.75, 5.25], v=8.0, omega=0.05)
side = {"prev": synthetic_gt(scene.step(-0.5), m.grid), "cur": synthetic_gt(scene, m.grid), "next": synthetic_gt(scene.step(0.5), m.grid)}
b["gt"] = [side] * 2
b["gap_steps"] = [{"prev": [(-0.5, 8.0, 0.05)], "next": [(0.5, 8.0, 0.05)]}] * 2
cfg = T.TrainConfig(device="cpu", gap_ego=True, radar_aux=1.0)
res = {}
for tt in (True, False):
    o, r, _ = T.forward_loss(m, copy.deepcopy(loss), copy.deepcopy(b), 0.3, "cpu", "off", cfg, train_terms=tt)
    res[f"fl_{tt}"] = {k: v.detach().clone() if torch.is_tensor(v) else v for k, v in o.items() if not isinstance(v, dict)}
    res[f"fl_{tt}_keys"] = list(o.keys()); res[f"fl_{tt}_reskeys"] = sorted(r.keys())
acc, ok, _, totals = T.run_clip(m, copy.deepcopy(loss), [copy.deepcopy(b)], 0.3, "cpu", "off", cfg)
res["clip"] = (list(acc.items()), ok, totals)
print({k: (len(v) if hasattr(v, '__len__') else v) for k, v in res.items()})
torch.save(res, out)
