"""critic probe 2 (E372): where the float32 / float64 gap of the live input gradient comes from; scratch, no record"""
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


def grads(mod, batch, dtype, step, loss_kw):
    torch.manual_seed(0)
    m = mod.JTokenHead(live_inputs=True).to(dtype)
    m.set_step(step)
    m.eval()
    x = {k: (v.to(dtype).clone().requires_grad_(True) if v.is_floating_point() else v) for k, v in batch.items()}
    loss = mod.jtoken_loss(m(x), x, **loss_kw)[0]
    loss.backward()
    ig = {k: x[k].grad.double() for k in JK.INPUTS if x[k].is_floating_point()}
    pg = {n: p.grad.double() for n, p in m.named_parameters() if p.grad is not None}
    return ig, pg, float(loss)


def rel(a, r):
    return float((a - r).norm() / r.norm().clamp_min(1e-300))


for name, bb in (("3 x 64", JK.synthetic_batch(3, 64, seed=1)), ("2 x 384", JK.synthetic_batch(2, 384, seed=2))):
    for step in (0, JK.SWITCH + 5):
        for loss_kw in (dict(none_weight=0.25), dict(none_weight=0.25, scale_w=0.0)):
            i32, p32, l32 = grads(JK, bb, torch.float32, step, loss_kw)
            i64, p64, l64 = grads(D64, bb, torch.float64, step, loss_kw)
            pw = sorted(((rel(p32[n], p64[n]), n) for n in p64), reverse=True)
            print(f"{name} step {step} {loss_kw}: loss {l32:.6f} vs {l64:.6f}; input grads rel err " + ", ".join(f"{k} {rel(i32[k], i64[k]):.1e}" for k in ("tok", "pos", "dirv", "e"))
                  + f"; param grads rel err: median {sorted(v for v, _ in pw)[len(pw) // 2]:.1e}, worst " + ", ".join(f"{n} {v:.1e}" for v, n in pw[:3]))
