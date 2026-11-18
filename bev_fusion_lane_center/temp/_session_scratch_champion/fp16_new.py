import os, sys, torch
sys.path.insert(0, "training/lanegraph")
from lg_model import LaneGraphNet
net = LaneGraphNet.from_runs(0)
path = "results/Ergebnis_Spurmitte_2026-11-15/champion_weights_fp16.pt"
sd = {k: (v.half() if v.is_floating_point() else v) for k, v in net.state_dict().items()}
torch.save(dict(state=sd, note="training/lanegraph LaneGraphNet state dict of seed 0 in float16; load: net = LaneGraphNet(); net.load_state_dict({k: v.float() for k, v in ck['state'].items()}); the float32 originals are runs/e273_p_s0/model.pt + runs/e324_s_s0/model.pt"), path)
h = LaneGraphNet(); h.load_state_dict({k: v.float() for k, v in torch.load(path)["state"].items()})
print("bytes", os.path.getsize(path), "params", sum(p.numel() for p in net.parameters()))
