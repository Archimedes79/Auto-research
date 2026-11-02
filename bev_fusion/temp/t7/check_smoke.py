"""t7 scratch: the smoke checkpoint stores the lane branch, its mix left zero after 2 steps, frozen res4 unchanged."""
import glob
import sys

import torch

run = sys.argv[1] if len(sys.argv) > 1 else "temp/t7/runs/smoke_res4"
ck = sorted(glob.glob(f"{run}/ckpt_*.pt"))[-1]
sd = torch.load(ck, map_location="cpu", weights_only=False)["state"]
lane = {k: v for k, v in sd.items() if k.startswith("model.lane.")}
print(ck, len(lane), "lane tensors")
print("mix weight abs sum", float(sd["model.lane.mix.weight"].abs().sum()))
ref = torch.load("models/pretrained/dualbev-r50-cbgs.pth", map_location="cpu", weights_only=True, mmap=True)
ref = ref.get("state_dict", ref)
k = "model.lane.features.layer1.0.conv1.weight"
if k in sd:
    print("res4 layer1 unchanged:", torch.equal(sd[k], ref["img_backbone.layer1.0.conv1.weight"]))
