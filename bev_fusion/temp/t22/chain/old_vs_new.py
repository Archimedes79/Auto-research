"""scratch: the option OFF reproduces the code before the student, with the lane branch on (the critic's idea): the SAME seeded model and batch through the training/ of commit a8f184a4 (the last commit
before the student; extracted to temp/t22/chain/old/training) and through the live training/, outputs compared bit for bit.
    python temp/t22/chain/old_vs_new.py new|old|compare"""
import os, sys
import torch
torch.set_num_threads(2)
mode = sys.argv[1]
ROOT = os.getcwd()
OUT = os.path.join(ROOT, "temp", "t22", "chain", "ovn")
os.makedirs(OUT, exist_ok=True)
sys.path[:0] = [os.path.join(ROOT, "temp", "t22", "chain", "old", "training") if mode == "old" else os.path.join(ROOT, "training"), os.path.join(ROOT, "tests")] if mode != "compare" else []
if mode == "compare":
    bad = 0
    for name in ("window", "pair"):
        a, b = torch.load(os.path.join(OUT, f"{name}_old.pt")), torch.load(os.path.join(OUT, f"{name}_new.pt"))
        same = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
        nz = sum(float(v.abs().max()) > 0 for v in a.values())
        print(f"{name}: {len(a)} outputs, {nz} non-zero, old == new bit for bit: {same}", "" if same else [k for k in a if not torch.equal(a[k], b[k])][:5])
        bad += not same
    sys.exit(bad)
from feature_cache import CachedBEVFusionModel
from model import ModelConfig
GRID = {"x_max": 16.0, "y_min": -8.0, "y_max": 8.0}
KW = dict(pretrained="", trunk_pretrained="", grid_kwargs=GRID, oversample=1, c_cam=16, c_rad=8, c_fuse=24, trunk_width=0.25, p_drop=0.0, dense_outputs=True, camera_features="dual", lane_branch="res4")
for name, extra, Fn in (("window", {}, 3), ("pair", {"cam_buffer": "pair"}, 2)):
    bpath = os.path.join(OUT, f"{name}_batch.pt")
    if mode == "new":
        from test_slots import batch_for
        b = batch_for(ModelConfig(**dict(KW, **extra)), B=2, Fn=Fn)
        b["lane_img"] = torch.randint(0, 256, (2, 3, 256, 704), dtype=torch.uint8, generator=torch.Generator().manual_seed(2))
        torch.save(b, bpath)
    b = torch.load(bpath, weights_only=False)
    torch.manual_seed(0)
    m = CachedBEVFusionModel(ModelConfig(**dict(KW, **extra))).eval()
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():                                          # the lane branch starts at zero: give it a contribution
        for q in m.lane.mix.parameters():
            q.copy_(torch.randn(q.shape, generator=g) * 0.1)
        res = m(b, dts=(0.0, 0.5))
    out = {"features": res["features"]}
    for dt, p in res["out"].items():
        for k, v in p.items():
            for kk, t in (v.items() if isinstance(v, dict) else [("", v)]):
                if torch.is_tensor(t):
                    out[f"{dt}/{k}/{kk}"] = t
    for branch, st in res["state"].items():
        for k, v in (st or {}).items():
            if torch.is_tensor(v):
                out[f"state/{branch}/{k}"] = v
    torch.save({k: v.detach().cpu() for k, v in out.items()}, os.path.join(OUT, f"{name}_{mode}.pt"))
    print(mode, name, len(out), "outputs; lane contribution max", float(m.lane.mix.weight.abs().max()))
