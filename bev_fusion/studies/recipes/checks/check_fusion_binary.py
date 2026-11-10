"""Binary-data test of the camera-radar fusion (training/fusion.py) in front of the BEV trunk and head.

World: a binary BEV truth (random rectangles) on a 64 x 56 grid, x forward (row index i = range). Two sensors observe it:
  camera planes [obs, conf]: obs = truth with each cell flipped with probability q(i); conf = 1 - q(i) (the confidence
                             channel, like the mass of the plan). "informative": q grows with range, 0.03 near to 0.47 far;
                             "noise": q = 0.5 everywhere (conf 0.5)
  radar planes [hit]:        sparse hits: a truth cell is detected with probability 0.35, a false hit with 0.01 per cell
                             ("informative", the same at every range); "clutter": hits at random cells, 5 %, unrelated
No coordinate channels: the network never sees where a cell is except through the confidence channels (the strip rule).
Scenarios: S1 camera informative + radar clutter, S2 camera noise + radar informative, S3 both informative
(complementary: camera near, radar far). Fusions: ConcatFusion, GateFusion (+ modality dropout 0.3) -> BEVTrunk
(width 1/4, light up, 1 class). Test conditions on a fixed set: both, camera off (planes zeroed), radar off,
radar shuffled (hits of another sample). IoU of the positive class at 0.5, near third / far third of the rows.
Run from bev_fusion: wsl ~/miniconda3/envs/gcar/bin/python temp/checks/check_fusion_binary.py [steps] [batch] [device]
"""
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "training")
from bev_trunk import BEVTrunk  # noqa: E402
from fusion import ConcatFusion, GateFusion  # noqa: E402

X, Y = 64, 56
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 250
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 8
DEV = sys.argv[3] if len(sys.argv) > 3 else "cpu"
torch.set_num_threads(max(1, torch.get_num_threads()))


def make_truth(g, B):
    t = torch.zeros(B, 1, X, Y)
    for b in range(B):
        for _ in range(int(torch.randint(3, 7, (1,), generator=g))):
            h, w = int(torch.randint(2, 8, (1,), generator=g)), int(torch.randint(2, 8, (1,), generator=g))
            i, j = int(torch.randint(0, X - h, (1,), generator=g)), int(torch.randint(0, Y - w, (1,), generator=g))
            t[b, 0, i:i + h, j:j + w] = 1
    return t


def make_batch(g, B, scenario):
    truth = make_truth(g, B)
    rows = torch.arange(X, dtype=torch.float32).view(1, 1, X, 1)
    if scenario == "S2":                                   # camera is noise
        q = torch.full((1, 1, X, 1), 0.5)
    else:
        q = 0.03 + 0.44 * (rows / (X - 1)) ** 1.5
    flip = (torch.rand(B, 1, X, Y, generator=g) < q).float()
    obs = (truth + flip) % 2
    cam = torch.cat([obs, (1 - q).expand(B, 1, X, Y)], 1)
    if scenario == "S1":                                   # radar is clutter
        hit = (torch.rand(B, 1, X, Y, generator=g) < 0.05).float()
    else:
        p = torch.where(truth > 0, torch.tensor(0.35), torch.tensor(0.01))
        hit = (torch.rand(B, 1, X, Y, generator=g) < p).float()
    return cam, hit, truth


class Net(torch.nn.Module):
    def __init__(self, fusion):
        super().__init__()
        self.fusion = (ConcatFusion if fusion == "concat" else GateFusion)(2, 1, 24, p_drop=0.3)
        self.trunk = BEVTrunk(cin=24, n_classes=1, width=0.25, light_up=True)

    def forward(self, cam, rad):
        return self.trunk(self.fusion(cam, rad))


def iou(logits, truth, rows):
    pred = (logits > 0).float()[:, :, rows]
    t = truth[:, :, rows]
    inter, union = (pred * t).sum().item(), ((pred + t) > 0).float().sum().item()
    return inter / max(1.0, union)


def evaluate(net, test):
    cam, rad, truth = test
    net.eval()
    conds = {"both": (cam, rad), "camera off": (torch.zeros_like(cam), rad), "radar off": (cam, torch.zeros_like(rad)),
             "radar shuffled": (cam, rad.roll(1, 0))}
    near, far = slice(0, X // 3), slice(2 * X // 3, X)
    out = {}
    with torch.no_grad():
        for name, (c, r) in conds.items():
            lg = net(c.to(DEV), r.to(DEV)).cpu()
            out[name] = (iou(lg, truth, near), iou(lg, truth, far))
    return out


def run(scenario, fusion, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = Net(fusion).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    t0 = time.perf_counter()
    for step in range(STEPS):
        net.train()
        cam, rad, truth = make_batch(g, BATCH, scenario)
        loss = F.binary_cross_entropy_with_logits(net(cam.to(DEV), rad.to(DEV)), truth.to(DEV),
                                                  pos_weight=torch.tensor(3.0, device=DEV))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 9:
            print(f"   ({(time.perf_counter() - t0) / 10:.2f} s per step on {DEV})", flush=True)
    test = make_batch(torch.Generator().manual_seed(123), 96, scenario)
    res = evaluate(net, test)
    gates = None
    if fusion == "gate":
        net.eval()
        cam, rad, _ = test
        with torch.no_grad():
            sw = net.fusion.gates(cam.to(DEV), rad.to(DEV)).cpu()          # (B, 2, 1, X, Y)
        per_row = sw[:, 1, 0].mean((0, 2))                                  # radar -> camera stream, mean per row
        gates = (per_row[:X // 3].mean().item(), per_row[2 * X // 3:].mean().item())
    return res, loss.item(), gates


if __name__ == "__main__":
    names = {"S1": "S1 camera informative, radar clutter", "S2": "S2 camera noise, radar informative",
             "S3": "S3 both informative (camera near, radar far)"}
    print(f"{STEPS} steps, batch {BATCH}, grid {X} x {Y}; IoU near third / far third of the rows")
    print(f"{'scenario':46s} {'fusion':7s} {'both':>11s} {'camera off':>11s} {'radar off':>11s} {'radar shuffled':>15s}   gate radar->camera near / far")
    for sc in ("S1", "S2", "S3"):
        for fusion in ("concat", "gate"):
            res, loss, gates = run(sc, fusion)
            cells = " ".join(f"{res[k][0]:.2f}/{res[k][1]:.2f}".rjust(11 if k != "radar shuffled" else 15) for k in res)
            gtxt = f"{gates[0]:.2f} / {gates[1]:.2f}" if gates else "-"
            print(f"{names[sc]:46s} {fusion:7s} {cells}   {gtxt}", flush=True)
