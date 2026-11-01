"""Layer-0 toy tests of the Poisson / PHD set loss (training/losses/p16_loss.py phd_loss, K193, E244): cases with a known answer.

    wsl bash tools/env/_py14.sh temp/loop19/test_phd.py        (CPU)

1. one edge of 10 m, pi = 1, ten points on it: L = (mass - sum log(lambda + eps)) / n with the closed form of the mixture at its centre
2. no valid edge: L = -log(eps) = 6.908 per point
3. two identical edges: the optimum of pi_1 + pi_2 is 1 (each 0.5), found on a grid
4. gradient is non-zero for a point 6 m from a pi = 0.1 edge (the hinge recall term gives zero there: dead zone)
5. positions get a gradient; invalid (pi = 0 / masked) edges contribute nothing; batch of two frames equals the mean of the single frames
Prints PASS / FAIL.
"""
import math
import sys

import torch

import p16_loss as PL

S, G, MIX, EPS = PL.PHD_SIGMA, PL.PHD_GAMMA, PL.PHD_MIX, PL.PHD_EPS
k0 = MIX / (S * math.sqrt(2 * math.pi)) + (1 - MIX) / (math.pi * G)  # density at distance 0
ok = True


def check(name, cond, info=""):
    global ok
    ok &= bool(cond)
    print(f"  {'ok  ' if cond else 'FAIL'} {name} {info}")


def frame(edges, pts):
    """edges: list of (ax, ay, bx, by, pi); node 0 of each edge pair: parents are separate root nodes. returns res, b for B=1"""
    n = 2 * len(edges)
    pos = torch.zeros(1, n, 2)
    par = -torch.ones(1, n, dtype=torch.long)
    pi = torch.zeros(1, n)
    valid = torch.zeros(1, n)
    for e, (ax, ay, bx, by, p) in enumerate(edges):
        pos[0, 2 * e], pos[0, 2 * e + 1] = torch.tensor([ax, ay]), torch.tensor([bx, by])
        par[0, 2 * e + 1] = 2 * e
        pi[0, 2 * e + 1], valid[0, 2 * e + 1] = p, 1.0
    g = torch.tensor(pts, dtype=torch.float32)[None]  # every 2nd point is used: duplicate each so that [::2] gives the list
    g05 = torch.repeat_interleave(g, 2, dim=1)
    return dict(pos=pos, parent=par, pi=pi, valid=valid), dict(g05=g05, m05=torch.ones(1, g05.shape[1]))


pts10 = [(0.0, float(y)) for y in range(10)]
res, b = frame([(0, 0, 0, 10, 1.0)], pts10)
L1 = float(PL.phd_loss(res, b))
want = (10.0 - 10 * math.log(k0 + EPS)) / 10
check("1 one edge, ten points on it", abs(L1 - want) < 1e-4, f"L {L1:.5f} closed form {want:.5f}")

res, b = frame([(0, 0, 0, 10, 0.0)], pts10)
res["valid"][:] = 0
L2 = float(PL.phd_loss(res, b))
check("2 no valid edge", abs(L2 + math.log(EPS)) < 1e-4, f"L {L2:.4f} -log(eps) {-math.log(EPS):.4f}")

grid = [i / 100 for i in range(5, 100, 5)]
vals = []
for p in grid:
    res, b = frame([(0, 0, 0, 10, p), (0, 0, 0, 10, p)], pts10)
    vals.append(float(PL.phd_loss(res, b)))
pbest = grid[vals.index(min(vals))]
check("3 two identical edges share the mass", abs(pbest - 0.5) <= 0.05, f"best pi each {pbest} (sum {2 * pbest:.2f})")

res, b = frame([(0, 0, 0, 10, 0.1)], [(6.0, 5.0)])
res["pi"].requires_grad_(True)
res["pos"].requires_grad_(True)
PL.phd_loss(res, b).backward()
g_pi = float(res["pi"].grad[0, 1])
check("4 gradient 6 m from a pi = 0.1 edge", abs(g_pi) > 1e-4 and math.isfinite(g_pi), f"d loss / d pi = {g_pi:.4f}")
check("5a positions get a gradient", float(res["pos"].grad.abs().sum()) > 0)

res, b = frame([(0, 0, 0, 10, 1.0), (0, 0, 0, 10, 0.0)], pts10)
res2, b2 = frame([(0, 0, 0, 10, 1.0)], pts10)
check("5b a pi = 0 edge contributes nothing", abs(float(PL.phd_loss(res, b)) - float(PL.phd_loss(res2, b2))) < 1e-5)

resA, bA = frame([(0, 0, 0, 10, 1.0)], pts10)
resB, bB = frame([(0, 0, 0, 10, 0.5)], pts10[:6])
cat = lambda x, y: {k: torch.cat([x[k], y[k]]) for k in x}  # noqa: E731
n = max(resA["pos"].shape[1], resB["pos"].shape[1])
resB = {k: (torch.nn.functional.pad(v, (0, 0, 0, n - v.shape[1])) if v.dim() == 3 else torch.nn.functional.pad(v, (0, n - v.shape[1]), value=-1 if k == "parent" else 0)) for k, v in resB.items()}
gA, gB = bA["g05"], bB["g05"]
m = max(gA.shape[1], gB.shape[1])
bAB = dict(g05=torch.cat([torch.nn.functional.pad(gA, (0, 0, 0, m - gA.shape[1]), value=1e4), torch.nn.functional.pad(gB, (0, 0, 0, m - gB.shape[1]), value=1e4)]),
           m05=torch.cat([torch.nn.functional.pad(bA["m05"], (0, m - gA.shape[1])), torch.nn.functional.pad(bB["m05"], (0, m - gB.shape[1]))]))
both = float(PL.phd_loss(cat(resA, resB), bAB))
single = 0.5 * (float(PL.phd_loss(*frame([(0, 0, 0, 10, 1.0)], pts10))) + float(PL.phd_loss(*frame([(0, 0, 0, 10, 0.5)], pts10[:6]))))
check("5c batch of two frames = mean of the single frames", abs(both - single) < 1e-4, f"{both:.5f} vs {single:.5f}")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
