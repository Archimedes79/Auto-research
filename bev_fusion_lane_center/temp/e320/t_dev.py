"""E317 scratch: tests of the DEV copies (temp/e317/dev) before they replace the live files.
usage: wsl bash tools/env/_py14.sh e317/t_dev.py <selftest | regress | real> [orig]
  selftest  the module self-tests of the dev jends.py and qjunction.py
  regress   flag-off regression: the ORIGINAL (temp/e317/orig) and the DEV jends.py on the same synthetic batch, bit-identical state / outputs / loss / gradients
  real      real frames on CPU: the trained E312 model (runs/e312_p_s0) through orig and dev, outputs bit-identical; the geometry loss of E312's own curves on real train frames (the starting point of E317)
"""
import importlib
import os
import sys

import torch

torch.set_num_threads(2)
ROOT = os.environ["BFLC_ROOT"]
HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("training", "training/grid", "training/data", "training/data/prep", "training/train", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
which = sys.argv[2] if len(sys.argv) > 2 else "dev"
sys.path.insert(0, os.path.join(HERE, which))                                  # the dev (or orig) copies first; modules they import come from the live folders


def load(name, folder):
    """import module `name` from temp/e317/<folder> under the alias name_<folder> (and under `name` when it is the first)"""
    import importlib.util
    from importlib.machinery import SourceFileLoader                           # the copies are named *.pysrc: a second jends.py in temp/ makes `_py14.sh <name>.py` ambiguous and breaks the chains
    ld = SourceFileLoader(f"{name}_{folder}", os.path.join(HERE, folder, name + ".pysrc"))
    spec = importlib.util.spec_from_loader(ld.name, ld)
    m = importlib.util.module_from_spec(spec)
    ld.exec_module(m)
    return m


def synth():
    torch.manual_seed(0)
    B, d = 2, 64
    jout = dict(j1=torch.randn(B, 8, 100, 100) * 0.1 - 2.0, j05=torch.randn(B, 2, 200, 200) - 2.0, f1=torch.randn(B, d, 100, 100))
    jout["j1"][:, 4], jout["j1"][:, 6] = 0.0, 0.0
    jout["j1"][:, 5], jout["j1"][:, 7] = 5.0, 5.0
    P = dict(centre1=torch.randn(B, 12, 100, 100))
    ent = torch.tensor([[[-1.75, -8.0, 0.0, 1.0], [1.75, -8.0, 0.0, 1.0]]] * B)
    ext = torch.tensor([[[-1.75, 8.0, 0.0, 1.0], [1.75, 8.0, 0.0, 1.0]]] * B)
    gt = dict(ent=ent, ext=ext, ent_m=torch.ones(B, 2, dtype=torch.bool), ext_m=torch.ones(B, 2, dtype=torch.bool))
    batch = dict(ent=ent, ext=ext, ent_m=gt["ent_m"], ext_m=gt["ext_m"], pairs=torch.tensor([[[0, 0], [1, 1]]] * B), pdrv=torch.ones(B, 2), pscale=torch.full((B, 2, 2), 0.3))
    return jout, P, gt, batch


def run_model(J, scales, kw):
    jout, P, gt, batch = synth()
    torch.manual_seed(1)
    m = J.JEnds(d=64, order=True, evidence=True, scales=scales, **kw)
    jr = dict(jout, j1=jout["j1"].clone().requires_grad_(True))
    out = m(jr, P, gt=gt)
    loss, info = J.jends_loss(out, batch, driven_w=2.0)
    loss.backward()
    return m, out, loss, info, jr["j1"].grad


if sys.argv[1] == "selftest":
    JD = load("jends", os.environ.get("JFOLDER", "dev"))
    JD._selftest()
    QJ = load("qjunction", "dev")
    QJ._selftest()
    print("selftests ok")
elif sys.argv[1] == "regress":
    JO, JD = load("jends", "orig"), load("jends", "dev")
    for scales in (False, True):
        mo, oo, lo, io, go = run_model(JO, scales, {})
        md, od, ld, idv, gd = run_model(JD, scales, {})
        so, sd = mo.state_dict(), md.state_dict()
        assert list(so) == list(sd) and all(torch.equal(so[k], sd[k]) for k in so), "the parameters (init) differ"
        for k in oo:
            if torch.is_tensor(oo[k]):
                assert torch.equal(oo[k], od[k]), f"output {k} differs"
        assert set(oo) == set(od), (set(oo) ^ set(od))
        assert torch.equal(lo, ld) and io == idv, (float(lo), float(ld), io, idv)
        assert torch.equal(go, gd), "the gradient into the heat differs"
        for (n, p), (_, q) in zip(mo.named_parameters(), md.named_parameters()):
            assert torch.equal(p.grad, q.grad), f"gradient of {n} differs"
        print(f"regression scales={scales}: parameters, outputs, loss {float(lo):.6f}, info and every gradient identical")
    print("regress ok")
elif sys.argv[1] == "real":
    raise SystemExit("see t_real.py")
