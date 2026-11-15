import os, sys, torch
from champion.lanegraph import LaneGraphNet, FeatureStore, split_rows, CHAMPION
from champion.lanegraph.model import frames
from champion.lanegraph.readout import deliver
torch.set_grad_enabled(False)
net = LaneGraphNet.from_runs(0).eval()
for n, m in net.named_children():
    print(f"{n:10s} {sum(p.numel() for p in m.parameters()):>9d}")
print("total", sum(p.numel() for p in net.parameters()))
out_dir = sys.argv[1]
path = os.path.join(out_dir, "champion_weights_fp16.pt")
sd = {k: (v.half() if v.is_floating_point() else v) for k, v in net.state_dict().items()}
torch.save(dict(state=sd, note="LaneGraphNet (champion/lanegraph) state dict of seed 0 in float16; load with net.load_state_dict({k: v.float() for k, v in ck['state'].items()}); the float32 originals are runs/e273_p_s0/model.pt + runs/e324_s_s0/model.pt"), path)
print("bytes", os.path.getsize(path))
# what float16 weights change on 8 frames
h = LaneGraphNet().eval()
h.load_state_dict({k: v.float() for k, v in torch.load(path)["state"].items()})
x = FeatureStore()(split_rows("val")[:8])
a, b = net(x), h(x)
print("fp16 weights: max |d pair logit|", float((a["pairs"]["logit"] - b["pairs"]["logit"]).abs().max()), "max |d link logit| (same candidates:", a["lanes"]["pb"].shape == b["lanes"]["pb"].shape, ")")
same = 0
for (la, pa), (lb, pb) in zip(frames(a), frames(b)):
    ga, gb = deliver(la, pa, CHAMPION), deliver(lb, pb, CHAMPION)
    same += (len(ga.lines) == len(gb.lines) and ga.edges == gb.edges)
print("fp16 weights: frames with the same graph structure", same, "/ 8")
