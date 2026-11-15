"""critic: peak host memory of the temporal-block part of ONE 3-keyframe BPTT clip, window (3 calls, K=2) vs recurrent (9 calls, K=1); B=1, C given.
Mirrors train.run_clip: a loss per keyframe, backward(retain_graph) for t < L-1, the state carried in the graph, `out` of the previous keyframe alive
until the next forward returned. Peak = ru_maxrss minus the baseline after the model is built."""
import os, resource, sys
import torch
torch.set_num_threads(1)
ROOT = "/mnt/c/Auto-research/bev_fusion"
sys.path.insert(0, ROOT + "/training")
from lift_pd import Grid
from temporal_fusion import TemporalFusion

mode, C = sys.argv[1], int(sys.argv[2])
B = 1
grid = Grid()
X, Y = grid.nx, grid.ny
torch.manual_seed(0)
mod = TemporalFusion(C, C, grid).train()
sweep = (torch.full((B,), 1 / 12), torch.full((B,), 9.0), torch.full((B,), 0.1))
gap = (torch.full((B,), 0.5), torch.full((B,), 9.0), torch.full((B,), 0.1))
hop = (torch.full((B,), 0.167), torch.full((B,), 9.0), torch.full((B,), 0.1))
obs = torch.ones(B, 1, X, Y)
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
frames = lambda: torch.randn(B, C, X, Y, requires_grad=True)
state, fused = None, None
L = 3
outs = None
for t in range(L):
    if mode == "window":
        cur, p1, p2 = frames(), frames(), frames()
        past = [(p2, [sweep] * 4), (p1, [sweep] * 2)]
        out, state, _ = mod(cur, past, state, gap if t else None, obs)
    else:
        for k in (2, 1, 0):
            cur = frames()
            move = [hop] if (k == 2 and t) else [sweep] * 2
            past = [] if fused is None else [(fused, move)]
            out, state, _ = mod(cur, past, state, move if (k < 2 or t) else None, obs)
            fused = out
    loss = out.mean()
    loss.backward(retain_graph=t < L - 1)
    outs = (out, loss)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"{mode:9s} C={C}: base {base:.0f} MB, peak {peak:.0f} MB, delta {peak - base:.0f} MB (B=1)")
