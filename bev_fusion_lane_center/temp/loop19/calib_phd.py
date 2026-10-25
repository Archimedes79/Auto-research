"""E244 calibration (review finding 1): gradient norm of the Poisson set loss (phd) against the hinge recall term (rec, weight 1 and 3) on the same batches.

    wsl bash tools/env/_py14.sh temp/loop19/calib_phd.py        (GPU, about 1 min, beside a running training: small)

The j1 start checkpoint (e123_D7_r3_j1), recipe config of E219.j1eq, 12 training batches of 8 frames, teacher-forcing share 0 and 0.5 (start and end of the warm-up),
no parameter update. Prints the gradient norms of w_rec * rec, w_phd * phd per batch and their ratio; the weight that makes phd * w match rec at weight 1 is
w = ||g_rec|| / ||g_phd|| (median over batches). The card started from phd 0.3 as a guess.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.environ["BFLC_ROOT"], "training", "train"))
import p16_common as C  # noqa: E402
import p16_data as PD  # noqa: E402
import p16_loss as PL  # noqa: E402
import p16_train as T  # noqa: E402

SETJ = ["reach=null", "spf2=false", "temporal=false", "stage_b=false", "stage_c=true", "b=3", "mcl=true", 'ver_mode="survival"', 'link_mode="learned"', 'pi_mode="min"',
        "detach_pos=false", 'dedup="geo"', 'ctx="global"', 'pattern="teardrop"', 'seeds="lattice"', "warm=true", "revisit=false", 'constr="L"', "D=1", "M=192", "K0=96",
        "seed_gscale=1.0", "tolerant=false", "local_r=12.0", "junc=true"]
torch.manual_seed(0)
net, cfg = T.build("D7", T.parse_set(SETJ))
net = net.cuda()
sd = torch.load(f"{C.OUT}/e123/e123_D7_r3_j1/model.pt", map_location="cuda")
net.load_state_dict({k: v for k, v in sd.items() if k in net.state_dict() and net.state_dict()[k].shape == v.shape}, strict=False)
ds, dl = PD.loader("train", bs=8, workers=2, with_prev=cfg.temporal)
params = [p for p in net.parameters() if p.requires_grad]
rows = []
for i, b in enumerate(dl):
    if i >= 12:
        break
    b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
    share = 0.5 if i % 2 == 0 else 0.0
    net.train()
    with torch.autocast("cuda", dtype=torch.float16):
        res = net(b["x"].float(), gt=b if share > 0 else None, share=share, prior=None)
    _, l_rec = PL.graph_losses(res, b)
    l_phd = PL.phd_loss(res, b)
    g_rec = torch.autograd.grad(l_rec, params, retain_graph=True, allow_unused=True)
    g_phd = torch.autograd.grad(l_phd, params, retain_graph=True, allow_unused=True)
    n = lambda g: float(torch.sqrt(sum((x.float() ** 2).sum() for x in g if x is not None)))  # noqa: E731
    rows.append((share, float(l_rec), float(l_phd), n(g_rec), n(g_phd)))
    print(f"batch {i:2} share {share:.1f}  rec {rows[-1][1]:.3f} phd {rows[-1][2]:.3f}  |g rec| {rows[-1][3]:.3f}  |g phd| {rows[-1][4]:.3f}  ratio rec/phd {rows[-1][3] / max(rows[-1][4], 1e-9):.3f}", flush=True)
r = np.array([x[3] / max(x[4], 1e-9) for x in rows])
print(f"median ratio |g rec| / |g phd| = {np.median(r):.3f} (min {r.min():.3f}, max {r.max():.3f}): phd weight matching rec weight 1 = {np.median(r):.2f}, matching rec weight 3 = {3 * np.median(r):.2f}; the card guess 0.3")
