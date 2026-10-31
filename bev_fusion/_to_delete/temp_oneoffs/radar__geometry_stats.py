"""One-off (2026-10-26): value ranges of the radar fields and of geometry residuals (range / direction against the cell, ray origin).
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/radar/geometry_stats.py
"""
import math

import torch

fr = torch.load("temp/radar/cache/mini_multisweep.pt")
c = {k: torch.cat([a[k] for a in fr]) for k in ["x", "y", "r", "psi", "dt", "vr", "sid", "rcs"]}
m = (c["dt"] <= 0.5) & (c["x"] >= 0) & (c["x"] < 60) & (c["y"] >= -26) & (c["y"] < 26)
c = {k: v[m] for k, v in c.items()}
x, y, r, psi = c["x"], c["y"], c["r"], c["psi"]
sx, sy = x - r * torch.cos(psi), y - r * torch.sin(psi)
cx, cy = (torch.floor(x / 0.5) + 0.5) * 0.5, (torch.floor((y + 26) / 0.5) + 0.5) * 0.5 - 26
rho, beta = torch.hypot(cx, cy), torch.atan2(cy, cx)
dr = r - rho
dpsi = torch.remainder(psi - beta + math.pi, 2 * math.pi) - math.pi
# residual against the bearing / distance of the cell centre seen from the ray origin (the physics: p = s + r u)
r_geo = torch.hypot(cx - sx, cy - sy)
psi_geo = torch.atan2(cy - sy, cx - sx)


def q(t):
    return [round(v, 2) for v in torch.quantile(t.float(), torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99])).tolist()]


print("n", len(x), "(quantiles 1 / 10 / 50 / 90 / 99 %)")
print("r", q(r))
print("ray origin x", q(sx), " y", q(sy))
print("r - |cell| from the vehicle origin", q(dr))
print("psi - bearing of the cell from the vehicle origin (deg)", q(torch.rad2deg(dpsi)))
print("r - |cell - origin| (exact origin)", q(r - r_geo))
print("psi - bearing of the cell from the origin (deg)", q(torch.rad2deg(torch.remainder(psi - psi_geo + math.pi, 2 * math.pi) - math.pi)))
print("vr", q(c["vr"]), " share |vr| > 0.5:", round((c["vr"].abs() > 0.5).float().mean().item(), 3))
print("rcs", q(c["rcs"]), " dt", q(c["dt"]), " distinct dt values (0.01 s):", torch.unique((c["dt"] * 100).round()).numel())
for s in range(5):
    k = c["sid"] == s
    if not k.any():
        continue
    print(f"sensor {s}: origin x {q(sx[k])[2]}, y {q(sy[k])[2]} (median), spread x p10-p90 {q(sx[k])[1]}..{q(sx[k])[3]}")
