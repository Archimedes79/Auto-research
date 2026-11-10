import os, sys, torch
ROOT = "/mnt/c/Auto-research/bev_fusion"
os.chdir(ROOT)
sys.path[:0] = [os.path.join(ROOT, "training"), ROOT, os.path.join(ROOT, "..")]
import train as T
from trainkit import checkpoint as ck
torch.set_num_threads(2)
path = "runs/t20_base_s0/ckpt_00006000.pt"
c = torch.load(path, map_location="cpu", weights_only=False)
print("ckpt keys: state", len(c["state"]), "ema", len(c["ema"]), "cfg.model has lane_use_depth:", "lane_use_depth" in c["cfg"]["model"])
res = {}
for name in ("t21_ctl", "t24_lane_pull"):
    import json, splits; raw = splits.resolve(json.load(open(f"training/configs/{name}.json"))); cfg = T.TrainConfig(**{k: v for k, v in raw.items() if not k.startswith("_")})
    model, loss, cached = T.build(cfg)
    fresh = ck.warm_start(path, model, loss, ck=c)
    print(name, "| lane.lift.use_depth:", model.lane.lift.use_depth, "| model tensors:", len(model.state_dict()), "| fresh keys (not in checkpoint):", fresh)
    res[name] = {k: v.clone() for k, v in model.state_dict().items()}
a, b = res["t21_ctl"], res["t24_lane_pull"]
print("same keys:", list(a) == list(b), "| every tensor equal after warm start:", all(torch.equal(a[k], b[k]) for k in a))
