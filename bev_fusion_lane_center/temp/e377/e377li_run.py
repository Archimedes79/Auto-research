"""throw-away (E372 live_inputs step): run the scratch copies temp/e377/e377li_{jtoken,jtoken_train,test_jtoken}.py in place of the live files.
    e377li_run.py [test names]     # the scratch test file
    e377li_run.py head             # the scratch head against HEAD's jtoken.py (e377li_head_jtoken.py): default and live, outputs and every parameter gradient bit for bit
"""
import importlib.util
import os
import runpy
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
for sub in ("evaluation", "training/train", "training/data/prep", "training/data", "training", "training/grid"):
    sys.path.insert(0, os.path.join(ROOT, sub))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


JK = load("jtoken", os.path.join(HERE, "e377li_jtoken.py"))
load("jtoken_train", os.path.join(HERE, "e377li_jtoken_train.py"))
if sys.argv[1:2] == ["head"]:
    import torch
    torch.set_num_threads(2)
    H = load("jtoken_head", os.path.join(HERE, "e377li_head_jtoken.py"))
    b = JK.synthetic_batch(3, 48, 0)
    for step in (0, JK.SWITCH + 1):
        def run(mod, **kw):
            torch.manual_seed(0)
            m = mod.JTokenHead(**kw)
            m.set_step(step)
            x = {k: (v.clone().requires_grad_(True) if v.is_floating_point() else v) for k, v in b.items()}
            out = m(x)
            mod.jtoken_loss(out, x, none_weight=0.25)[0].backward()
            return out, {n: p.grad for n, p in m.named_parameters()}, {k: x[k].grad for k in JK.INPUTS if x[k].is_floating_point()}
        ref, new, live = run(H), run(JK), run(JK, live_inputs=True)
        for name, r in (("default", new), ("live", live)):
            eq_out = all(torch.equal(ref[0][k], r[0][k]) for k in ("logp", "types", "types1", "scale", "x", "w", "cand", "chord", "valid"))
            eq_grad = all(torch.equal(g, r[1][n]) for n, g in ref[1].items())
            print(f"step {step} {name}: outputs = HEAD {eq_out}, all {len(ref[1])} parameter gradients = HEAD {eq_grad}, input grads: "
                  + ", ".join(f"{k} {'None' if g is None else f'{float(g.abs().sum()):.3e}'}" for k, g in r[2].items()))
elif sys.argv[1:2] == ["jpast"]:                                          # test_jpast_train.py's in-process modes on the scratch head: `jpast one` (the selftest), `jpast real` (one batch of the real chains)
    T = os.path.join(ROOT, "training", "grid", "test_jpast_train.py")
    sys.argv = [T] + (["one", os.path.join(ROOT, "training", "train", "jpast_train.py"), os.path.join(ROOT, "evaluation", "jpast_eval.py")] if sys.argv[2] == "one" else ["real"])
    runpy.run_path(sys.argv[0], run_name="__main__")
else:
    sys.argv = [os.path.join(HERE, "e377li_test_jtoken.py")] + sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")
