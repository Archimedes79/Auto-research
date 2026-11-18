"""throw-away (E372 live_inputs step, critic round): run the scratch copy temp/e377/e377li_test_jtoken.py against the live jtoken.py / jtoken_train.py.
    e377li_run.py [test names]     # the scratch test file
    e377li_run.py fd               # the analytic and finite-difference directional derivatives per (step, input) of jtoken (float64) and of the two FD mutants (a replica of `_live_fd_ok` without the assert)
"""
import os
import runpy
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
for sub in ("evaluation", "training/train", "training/data/prep", "training/data", "training", "training/grid"):
    sys.path.insert(0, os.path.join(ROOT, sub))
TEST = os.path.join(HERE, "e377li_test_jtoken.py")
if sys.argv[1:2] == ["fd"]:
    import torch
    t = runpy.run_path(TEST, run_name="e377li_test")
    JK, H = t["JK"], t["FD_H"]
    for name, edits in (("jtoken", []), ("geometry cut", [("JT.pair_geometry(pos, dirv, valid, e=w_in)", "JT.pair_geometry(pos.detach(), dirv.detach(), valid, e=w_in.detach())")]),
                        ("key weight cut", [("lw(w1 if learned else w_in)", "lw(w1 if learned else w_in.detach())")])):
        mod = t["_mutant"](*edits, double=True)
        b = {k: (v.double() if v.is_floating_point() else v) for k, v in JK.synthetic_batch(2, 64, seed=0).items()}
        gen, vm = torch.Generator().manual_seed(7), b["valid"].double()
        b["pos"] = b["pos"] + 0.01 * torch.randn(b["pos"].shape, generator=gen, dtype=torch.float64) * vm[..., None]
        keys = [k for k in JK.INPUTS if b[k].is_floating_point()]
        for step in (0, JK.SWITCH + 1):
            torch.manual_seed(0)
            m = mod.JTokenHead(live_inputs=True).to(torch.float64).eval()
            m.set_step(step)
            loss = lambda bb: mod.jtoken_loss(m(bb), bb, none_weight=0.25)[0]  # noqa: E731
            x = dict(b, **{k: b[k].clone().requires_grad_(True) for k in keys})
            loss(x).backward()
            row = []
            for k in keys:
                v = torch.randn(b[k].shape, generator=gen, dtype=torch.float64) * vm.view(*vm.shape, *[1] * (b[k].dim() - 2))
                with torch.no_grad():
                    fd = float(loss(dict(b, **{k: b[k] + H * v})) - loss(dict(b, **{k: b[k] - H * v}))) / (2 * H)
                an = float((x[k].grad * v).sum())
                row.append(f"{k} an {an:+.3e} err {abs(fd - an):.1e} rel {abs(fd - an) / max(abs(an), 1e-300):.1e}")
            print(f"{name} step {step}: " + "; ".join(row))
else:
    sys.argv = [TEST] + sys.argv[1:]
    runpy.run_path(TEST, run_name="__main__")
