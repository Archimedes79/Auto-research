"""critic probe for the E376 --pos-loss diff (read-only: imports jtoken / jtoken_train, builds HEAD's jtoken.py in memory)"""
import os, sys, subprocess, types, math
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
torch.set_num_threads(2)
ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/train", "training/data", "training/data/prep", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import jtoken as JK
import test_jtoken as TJ

def load_src(src, name):
    mod, path = types.ModuleType(name), list(sys.path)
    mod.__file__ = JK.__file__
    try:
        exec(compile(src, JK.__file__, "exec"), mod.__dict__)
    finally:
        sys.path[:] = path
    return mod

head_src = subprocess.check_output(["git", "show", "HEAD:bev_fusion_lane_center/training/grid/jtoken.py"], cwd=ROOT).decode()
OLD = load_src(head_src, "jtoken_head")

# 1. HEAD vs new on the head's real output: loss and every parameter gradient bit for bit (W 1 and 0.25), several batches and K
for K in (48, 384):
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        m = JK.JTokenHead()
        b = JK.synthetic_batch(2, K, seed=seed)
        out = m(b)
        ps = [p for p in m.parameters()]
        for w in (1.0, 0.25):
            lo, io = OLD.jtoken_loss(out, b, none_weight=w)
            for kw in ({}, {"pos_loss": "sum"}):
                ln, inn = JK.jtoken_loss(out, b, none_weight=w, **kw)
                go = torch.autograd.grad(lo, ps, retain_graph=True, allow_unused=True)
                gn = torch.autograd.grad(ln, ps, retain_graph=True, allow_unused=True)
                same_g = all((a is None and c is None) or torch.equal(a, c) for a, c in zip(go, gn))
                assert torch.equal(lo, ln) and io == inn and same_g, (K, seed, w, kw, float(lo), float(ln))
            lm, im = JK.jtoken_loss(out, b, none_weight=w, pos_loss="max")
            assert math.isfinite(float(lm)) and float(lm) > float(lo), (float(lm), float(lo))
print("1 HEAD == new sum (loss, info, all grads) bit for bit on K 48 / 384, seeds 0-2, W 1 / 0.25: ok")

# 2. _source_nll max against an independent loop, the none slot never in the max, one-member rows identical, all-masked rows finite
g = torch.Generator().manual_seed(5)
B, K = 3, 9
z = torch.randn(B, 2, K, K, generator=g) * 3
zn = torch.randn(B, 2, K, generator=g) * 3 + 4.0                 # the none slot often the largest
cand = torch.rand(B, 2, K, K, generator=g) < 0.5
cand[:, :, 0] = False                                             # source 0: no candidate at all (all-masked row)
y = torch.rand(B, K, K, generator=g) < 0.4
G = y[:, None] & cand
z = z.requires_grad_()
lp = JK.wiring_logp(z, zn, cand)
nm = JK._source_nll(lp, G, 0.25, "max")
ns = JK._source_nll(lp, G, 0.25, "sum")
ref = torch.empty_like(nm)
for bb in range(B):
    for c in range(2):
        for i in range(K):
            js = [j for j in range(K) if G[bb, c, i, j]]
            ref[bb, c, i] = -max(float(lp[bb, c, i, j]) for j in js) if js else -0.25 * float(lp[bb, c, i, K])
assert torch.allclose(nm, ref, atol=0, rtol=0), (nm - ref).abs().max()
one = G.sum(-1) == 1
assert one.any() and torch.equal(nm[one], ns[one]) and torch.equal(nm[~G.any(-1)], ns[~G.any(-1)])
none_big = (lp[..., K] > lp[..., :K].masked_fill(~G, -1e9).amax(-1)) & G.any(-1)
assert none_big.any(), "a G row where the none slot is the largest entry exists"
gr = torch.autograd.grad(nm.sum(), z)[0]
assert torch.isfinite(gr).all() and torch.isfinite(nm).all()
# the gradient of the positive term w.r.t. lp: one-hot on the argmax member of G, zero on the none slot and masked keys
lpl = lp.detach().requires_grad_()
glp = torch.autograd.grad(JK._source_nll(lpl, G, 0.25, "max").sum(), lpl)[0]
for bb in range(B):
    for c in range(2):
        for i in range(K):
            js = [j for j in range(K) if G[bb, c, i, j]]
            row = glp[bb, c, i]
            if js:
                jw = max(js, key=lambda j: float(lp[bb, c, i, j]))
                want = torch.zeros(K + 1); want[jw] = -1.0
                assert torch.equal(row, want), (bb, c, i, row, jw)
            else:
                want = torch.zeros(K + 1); want[K] = -0.25
                assert torch.equal(row, want), (bb, c, i, row)
print(f"2 max == independent loop exactly ({int(G.any(-1).sum())} G rows, {int(none_big.sum())} with the none slot above every member: not chosen), one-member rows == sum bit for bit ({int(one.sum())}), "
      f"empty rows == sum, all-masked source rows finite, d/dlp one-hot on the winning member (none slot and masked keys 0): ok")

# 3. ties: amax splits the gradient
lpt = torch.tensor([[[[-1.0, -1.0, -3.0, -0.5]]]], requires_grad=True)   # (1, 1, 1, K + 1), K = 3, members 0 and 1 tied
Gt = torch.tensor([[[[True, True, False]]]])
gt = torch.autograd.grad(JK._source_nll(lpt, Gt, 1.0, "max").sum(), lpt)[0]
print("3 exact tie of two members: d/dlp =", gt.flatten().tolist())

# 4. extra mutants against the committed test helpers
def grad_check(mod):
    out, b, _ = TJ._pos_out()
    def wl(o):
        return dict(o, logp=mod.wiring_logp(o["z"], o["zn"], o["cand"]))
    gz = {pos: torch.autograd.grad(mod.jtoken_loss(wl(out), b, none_weight=0.25, pos_loss=pos)[0], out["z"])[0][0, 0] for pos in mod.POS_LOSS}
    for i, (j1, j2) in ((0, (1, 2)), (3, (4, 5))):
        win, lose = (j1, j2) if out["z"][0, 0, i, j1] > out["z"][0, 0, i, j2] else (j2, j1)
        assert gz["sum"][i, win] < 0 and gz["sum"][i, lose] < 0 and gz["max"][i, win] < 0 and gz["max"][i, lose] > 0

def survives(name, edits, checks):
    mod = TJ._mutant(*edits)
    res = {}
    for cname, fn in checks.items():
        try:
            fn(mod)
            res[cname] = "PASSES (mutant survives)"
        except Exception as e:  # noqa: BLE001
            res[cname] = "fails (caught)"
    print(f"4 mutant '{name}': {res}")

checks = {"_check_pos max": lambda mod: TJ._check_pos("max", mod), "_check_pos sum": lambda mod: TJ._check_pos("sum", mod), "grad check": grad_check}
survives("none slot inside the max", [("g.amax(-1)", "torch.cat([g, lp[..., K:]], -1).amax(-1)")], checks)
survives("max detached (no gradient)", [("g.amax(-1)", "g.amax(-1).detach()")], checks)
survives("max over the sources axis", [("g.amax(-1)", "g.amax(-2)")], checks)
survives("max on the short class only", [("pos = -(torch.logsumexp(g, -1) if pos_loss == \"sum\" else g.amax(-1))",
                                           "pos = -torch.logsumexp(g, -1)\n    if pos_loss == \"max\":\n        pos = torch.cat([-g[:, :1].amax(-1), pos[:, 1:]], 1)")], checks)
survives("max with a soft temperature (logsumexp(10 g)/10)", [("g.amax(-1)", "torch.logsumexp(10 * g, -1) / 10")], checks)
survives("none weight also on the max term", [("pos = -(torch.logsumexp(g, -1) if pos_loss == \"sum\" else g.amax(-1))",
                                               "pos = -(torch.logsumexp(g, -1) if pos_loss == \"sum\" else none_weight * g.amax(-1))")], checks)
