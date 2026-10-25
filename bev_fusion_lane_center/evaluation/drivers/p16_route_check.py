"""FastRoute (CUDA graphs, padded) against the eager route head on stored token items: probabilities and flags."""
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p8_lib as L  # noqa: E402
import p10_net as PN  # noqa: E402
import p16_pipeline as PP  # noqa: E402
import sd_lib as SL  # noqa: E402

z = pickle.load(open(os.path.expanduser("~/exp_gcar_b123/p10/tok_p16_f2hr_val_clean.pkl"), "rb"))
sdl = L.cache("sd", z["part"])
net = PN.RouteXNet().cuda()
net.load_state_dict(torch.load(f"{L.OUT}/p16_f2hr_route_s0/model.pt", map_location="cuda"))
net.eval()
fr = PP.FastRoute(net)
dm, flips, n_ok, nn_ = 0.0, 0, 0, 0
for it in z["items"][:60]:
    row = it["row"]
    if sdl[row] is None or len(it["xy"]) == 0:
        continue
    sdm, _, _ = L.draw_sd(sdl[row], row, 0.0, 0.0, 1)
    ras = torch.from_numpy(SL.raster_sd(sdm))[None].cuda()
    tok = torch.from_numpy(it["tok"].astype(np.float32))[None].cuda()
    xy = torch.from_numpy(it["xy"])[None].cuda()
    cp = torch.from_numpy(it["cp"])[None].cuda()
    n = len(it["xy"])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        p0 = torch.sigmoid(net(tok, xy, cp, ras, torch.ones(1, n, dtype=torch.bool, device="cuda")).float())[0].cpu().numpy()
    p1 = fr(tok, xy, cp, ras, n).cpu().numpy()
    dm = max(dm, float(np.abs(p0 - p1).max()))
    flips += int(((p0 >= 0.5) != (p1 >= 0.5)).sum())
    nn_ += n
print(f"max |p_eager - p_graph| = {dm:.4f}; flag flips at 0.5: {flips} of {nn_} points")
