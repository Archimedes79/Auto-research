import os, sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion/training")
import torch
import losses as L
torch.manual_seed(0)
B, K, X, Y = 2, 2, 12, 14
lg = torch.randn(B, K, X, Y) * 3
tg = (torch.rand(B, K, X, Y) > 0.9).float(); tg[:, 1] = 0
dist = torch.where(tg > 0.5, torch.zeros_like(tg), torch.rand(B, K, X, Y) * 3)
w = torch.ones(B, 1, X, Y); w[0, :, :, :3] = 0
ref = L.line_bce_loss(lg, tg, dist, w)
print("fp32", ref)
for dt in (torch.bfloat16, torch.float16, torch.float64):
    try:
        out = L.line_bce_loss(lg.to(dt), tg, dist, w)
        print(dt, out, out.dtype)
    except Exception as e:
        print(dt, "ERR", str(e)[:120])
    try:
        out = L.line_bce_loss(lg.to(dt), tg.to(dt), dist, w)
        print(dt, "all-same-dtype", out, out.dtype)
    except Exception as e:
        print(dt, "all-same ERR", str(e)[:120])
    try:
        out = L.line_bce_loss(lg.to(dt), tg.to(dt), dist.to(dt), w.to(dt))
        print(dt, "all incl. w", out, out.dtype)
    except Exception as e:
        print(dt, "all incl w ERR", str(e)[:120])
# autocast bf16 on cpu
with torch.autocast("cpu", dtype=torch.bfloat16):
    out = L.line_bce_loss(lg, tg, dist, w)
    print("autocast cpu bf16 with fp32 inputs", out, out.dtype)
    lgb = (torch.randn(B, 2, 3, 3) @ torch.randn(3, 3)).sum() if False else lg.to(torch.bfloat16)
    out = L.line_bce_loss(lgb, tg, dist, w)
    print("autocast cpu bf16 with bf16 logits", out, out.dtype)
# all weights zero -> zero loss, finite zero gradient
lgg = lg.clone().requires_grad_(True)
o = L.line_bce_loss(lgg, tg, dist, torch.zeros(B, 1, X, Y)); o.sum().backward()
print("w=0 everywhere:", o, "grad max", lgg.grad.abs().max().item(), "finite", bool(torch.isfinite(lgg.grad).all()))
# a channel with no weighted cells (only band cells)
d2 = torch.full((B, K, X, Y), 0.5); t2 = torch.zeros(B, K, X, Y)
lgg = lg.clone().requires_grad_(True)
o = L.line_bce_loss(lgg, t2, d2, w); o.sum().backward()
print("only band cells:", o, "grad max", lgg.grad.abs().max().item())
# tiny weights (w = 1e-9 everywhere): denominator clamp 1e-6 vs sum ~ 1e-9*cells
o = L.line_bce_loss(lg, tg, dist, torch.full((B, 1, X, Y), 1e-9))
print("w=1e-9 everywhere:", o, " (ref fp32 mean-ish:", ref, ")")
# infinite logits with w=0 cell
lg3 = lg.clone(); lg3[0, 0, 0, 0] = float("inf")
o = L.line_bce_loss(lg3, tg, dist, w)
print("inf logit at a w=0 cell:", o)
lg3[0, 0, 5, 5] = float("inf")
o = L.line_bce_loss(lg3, tg, dist, w)
print("inf logit at a w>0 cell:", o)
