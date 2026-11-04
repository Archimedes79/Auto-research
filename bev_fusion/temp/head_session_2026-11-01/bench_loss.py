"""Where the time of the loss goes, on REAL GT frames and real-size tensors (CPU, 2 threads; loss forward + backward on leaf tensors, no optimiser step).
One micro-batch of the t11 / t10 recipe: 9 samples (batch 3 x clip 3), dt in (-0.5, 0, +0.5), the incumbent head layout (direction + hurdle), aux heads on.
On the GPU the tensor math below shrinks to launch overhead; the host parts (target building) cost the same: they are the critical path."""
import glob
import os
import statistics
import sys
import time
from dataclasses import replace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, os.path.dirname(ROOT))
torch.set_num_threads(2)
import losses as L                                              # noqa: E402
import object_head as O                                         # noqa: E402
from lift_pd import Grid                                        # noqa: E402

N = 9
files = sorted(glob.glob(os.path.join(ROOT, "data", "cache", "gt", "*_0.5_4_15efbde2d1.npz")))[::211][:N]
entries = [np.load(f, allow_pickle=True)["gt"].item() for f in files]
DTS = (-0.5, 0.0, 0.5)
gts = {dt: [e[{-0.5: "prev", 0.0: "cur", 0.5: "next"}[dt]] for e in entries] for dt in DTS}
grid = Grid()
X, Y = grid.nx, grid.ny
cfg = L.LossConfig(w_obj_vel=1.0)
lm = L.BEVLoss(cfg)
torch.manual_seed(0)
head = O.ObjectHead(8, groups=L.NUSC_GROUPS, direction=True, moving=True)


def preds():
    p = {}
    for dt in DTS:
        maps = {k: v.detach().clone().requires_grad_() for k, v in head(torch.randn(N, 8, X, Y)).items()}
        d = {"sem": torch.randn(N, len(lm.spec), X, Y, requires_grad=True), "obj": maps, "mass": torch.ones(N, 1, X, Y),
             "cam_aux": torch.randn(N, len(lm.cam_spec), X, Y, requires_grad=True),
             "radar_aux": torch.randn(N, len(lm.radar_spec), X, Y, requires_grad=True)}
        if dt == 0.0:
            d["v_hat"] = torch.randn(N, 2, X, Y, requires_grad=True)
        p[dt] = d
    return p


def med(fn, n=7):
    fn()
    ts = []
    for _ in range(n):
        a = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - a) * 1000)
    return statistics.median(ts)


print(f"{N} real frames x {len(DTS)} dt = {N * len(DTS)} target sets; grid {X} x {Y}; sem channels {len(lm.spec)}")
# 1 targets and weights from the GT (host side: depends on the GT only)
t_stack = med(lambda: [L.build_layer_targets(gts[dt], cfg, None, dt) for dt in DTS])
Ts = {dt: L.build_layer_targets(gts[dt], cfg, None, dt) for dt in DTS}
print(f"build_layer_targets  (stack ~25 layers per sample, weights): {t_stack:8.1f} ms per micro-batch")
# 2 object head targets + loss
out_maps = preds()
t_obj = med(lambda: [L.object_head_terms(out_maps[dt]["obj"], Ts[dt]["boxes"], Ts[dt]["w_obj"], grid, L.NUSC_GROUPS, cfg.w_obj_vel, cfg.max_obj, cfg.w_obj_mov) for dt in DTS])
t_bt = med(lambda: [O.build_targets([[L.head_box(b) for b in bs if L.head_box(b)] for bs in Ts[dt]["boxes"]], grid, L.NUSC_GROUPS, cfg.max_obj) for dt in DTS])
n_boxes = sum(len(b) for dt in DTS for b in Ts[dt]["boxes"])
print(f"object_head_terms    (box loop + Gaussian peaks + focal/L1): {t_obj:8.1f} ms   of which build_targets {t_bt:6.1f} ms   ({n_boxes} boxes over the 3 dt)")
# 3 map terms per head (tensor math)
sem = out_maps[0.0]["sem"]
w = {"area": Ts[0.0]["w_map"], "line": Ts[0.0]["w_lines"], "object": Ts[0.0]["w_obj"]}
t_map = med(lambda: L.map_terms(sem, Ts[0.0], lm.spec, w, cfg, 1.0))
print(f"map_terms (sem head, one dt, forward only):               {t_map:8.1f} ms")
# 4 whole BEVLoss forward and backward
pr = preds()
t_fwd = med(lambda: lm(preds_ := pr, gts, 1.0))
res = lm(pr, gts, 1.0)
terms = len(res)


def fb():
    p = preds()
    lm(p, gts, 1.0)["total"].backward()


t_fb = med(fb, 5)
t_p = med(preds, 5)
print(f"whole BEVLoss forward:                                    {t_fwd:8.1f} ms   ({terms} entries in the result dict)")
print(f"forward + backward (incl. making the leaf tensors {t_p:.0f} ms): {t_fb:8.1f} ms")
# 5 operator counts (a proxy for kernel launches on the GPU)
from torch.profiler import ProfilerActivity, profile            # noqa: E402
p = preds()
with profile(activities=[ProfilerActivity.CPU]) as prof:
    lm(p, gts, 1.0)["total"].backward()
n_ops = sum(e.count for e in prof.key_averages())
print(f"aten operator calls, forward + backward of the whole loss: {n_ops}")
tops = sorted(prof.key_averages(), key=lambda e: -e.self_cpu_time_total)[:6]
print("top self-CPU ops:", ", ".join(f"{e.key} {e.self_cpu_time_total / 1000:.0f} ms x{e.count}" for e in tops))
