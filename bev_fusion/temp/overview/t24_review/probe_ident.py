"""default-path identity: build + run models in tree argv[1], save everything to argv[2]"""
import os, sys, torch
tree, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(tree, "tests")); sys.path.insert(0, os.path.join(tree, "training"))
from feature_cache import CachedBEVFusionModel
from model import ModelConfig
from test_slots import batch_for
torch.set_num_threads(2)
GRID = {"x_max": 16.0, "y_min": -8.0, "y_max": 8.0}
BASE = dict(pretrained="", trunk_pretrained="", lane_pretrained="", grid_kwargs=GRID, c_cam=16, c_rad=8, c_fuse=24, trunk_width=0.25, p_drop=0.0)
CASES = {
 "dual_res4_os1": dict(camera_features="dual", lane_branch="res4", oversample=1),
 "dual_res4_os4": dict(camera_features="dual", lane_branch="res4", oversample=4),
 "dualbev_rgb_os2": dict(camera_features="dualbev", lane_branch="rgb", oversample=2),
 "dual_nolane": dict(camera_features="dual", lane_branch="", oversample=1),
 "dual_res4_rec": dict(camera_features="dual", lane_branch="res4", oversample=1, cam_buffer="recurrent"),
}
res = {}
for name, kw in CASES.items():
    for mode in ("eval", "train"):
        torch.manual_seed(0)
        m = CachedBEVFusionModel(ModelConfig(**dict(BASE, **kw)))
        if m.lane is not None:
            g = torch.Generator().manual_seed(3)
            with torch.no_grad():
                m.lane.mix.weight.copy_(torch.randn(m.lane.mix.weight.shape, generator=g) * 0.05)
                m.lane.mix.bias.copy_(torch.randn(m.lane.mix.bias.shape, generator=g) * 0.05)
        getattr(m, mode)()
        b = batch_for(m.cfg)
        b["lane_img"] = torch.randint(0, 256, (2, 3, 256, 704), dtype=torch.uint8, generator=torch.Generator().manual_seed(1))
        torch.manual_seed(5)
        r = m(b, dts=(0.0,))["out"][0.0]
        flat = {"sem": r["sem"].detach().clone()}
        flat.update({"obj." + k: v.detach().clone() for k, v in r["obj"].items()})
        res[f"{name}/{mode}"] = {"keys": list(m.state_dict().keys()), "sd": {k: v.clone() for k, v in m.state_dict().items()}, "out": flat}
torch.save(res, out)
print("saved", out, len(res))
