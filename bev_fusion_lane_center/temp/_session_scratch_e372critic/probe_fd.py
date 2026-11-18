"""critic probe 3 (E372): float32 vs float64 on a jittered batch (no gate ties at 8 m), and a float64 directional finite-difference check of the live input gradient with the geometry-cut mutant; scratch, no record"""
import os
import sys
import types

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch  # noqa: E402

torch.set_num_threads(2)
ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "training/data/prep", "training/train", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import jtoken as JK  # noqa: E402

SRC_WT = open(JK.__file__, encoding="utf-8").read()


def load(src, name):
    mod, path = types.ModuleType(name), list(sys.path)
    mod.__file__ = JK.__file__
    try:
        exec(compile(src, JK.__file__, "exec"), mod.__dict__)
    finally:
        sys.path[:] = path
    return mod


D64 = load(SRC_WT.replace(".float()", ".double()"), "jtoken_d64")
GEO = "g, allowed = JT.pair_geometry(pos, dirv, valid, e=w_in)"
CUT = load(SRC_WT.replace(".float()", ".double()").replace(GEO, "g, allowed = JT.pair_geometry(pos.detach(), dirv.detach(), valid, e=w_in.detach())"), "jtoken_cut")
LW = "x = layer(x, g, lw(w1 if learned else w_in), valid)"
CUT2 = load(SRC_WT.replace(".float()", ".double()").replace(LW, "x = layer(x, g, lw(w1 if learned else w_in.detach()), valid)"), "jtoken_cut2")


def jitter(b, s=0.01, seed=5):
    g = torch.Generator().manual_seed(seed)
    b = dict(b)
    b["pos"] = b["pos"] + s * torch.randn(b["pos"].shape, generator=g) * b["valid"][..., None]
    return b


def loss_of(mod, m, x):
    return mod.jtoken_loss(m(x), x, none_weight=0.25)[0]


def head(mod, dtype, step):
    torch.manual_seed(0)
    m = mod.JTokenHead(live_inputs=True).to(dtype)
    m.set_step(step)
    return m.eval()


def rel(a, r):
    return float((a - r).norm() / r.norm())


bb = jitter(JK.synthetic_batch(2, 64, seed=1))
for step in (0, JK.SWITCH + 5):
    out = {}
    for tag, mod, dt in (("f32", JK, torch.float32), ("f64", D64, torch.float64)):
        m = head(mod, dt, step)
        x = {k: (v.to(dt).clone().requires_grad_(True) if v.is_floating_point() else v) for k, v in bb.items()}
        loss = loss_of(mod, m, x)
        loss.backward()
        out[tag] = (float(loss), {k: x[k].grad.double() for k in JK.INPUTS if x[k].is_floating_point()})
    print(f"jittered 2 x 64, step {step}: loss f32 {out['f32'][0]:.7f} f64 {out['f64'][0]:.7f}; input grads f32 vs f64: " + ", ".join(f"{k} {rel(out['f32'][1][k], out['f64'][1][k]):.1e}" for k in out["f64"][1]))

# float64 directional derivative: analytic <grad, d> against the central difference, per input; the mutants must miss it
for step in (0, JK.SWITCH + 5):
    for tag, mod in (("live", D64), ("geometry cut", CUT), ("key weight cut", CUT2)):
        m = head(mod, torch.float64, step)
        base = {k: (v.double() if v.is_floating_point() else v) for k, v in bb.items()}
        x = {k: (v.clone().requires_grad_(True) if v.is_floating_point() else v) for k, v in base.items()}
        loss_of(mod, m, x).backward()
        res = []
        for k in ("pos", "dirv", "e", "tok"):
            g = torch.Generator().manual_seed(7)
            d = torch.randn(base[k].shape, generator=g, dtype=torch.float64) * (base["valid"][..., None] if base[k].dim() == 3 else base["valid"])
            h = 1e-6
            with torch.no_grad():
                lp = float(loss_of(mod, m, dict(base, **{k: base[k] + h * d})))
                lm = float(loss_of(mod, m, dict(base, **{k: base[k] - h * d})))
            fd, an = (lp - lm) / (2 * h), float((x[k].grad * d).sum())
            res.append(f"{k} fd {fd:.6e} an {an:.6e} rel {abs(fd - an) / max(abs(fd), 1e-30):.1e}")
        print(f"step {step} {tag}: " + "; ".join(res))
