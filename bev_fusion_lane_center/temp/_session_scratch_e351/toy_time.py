import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
R = os.environ["BFLC_ROOT"]
for sub in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.join(R, sub))
import torch
torch.set_num_threads(2)
print(torch.__version__)
import jends_train as JT
for kw in (dict(), dict(attach=True, scales=True)):
    t0 = time.time()
    m = JT.JEndModel(dict(d=64, ctx="none"), order=True, evidence=True, **kw).eval()
    x = torch.randn(1, 39, 200, 200)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        P, out, oe = m(x, gt=None)
    t1 = time.time()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        P2, out2, oe2 = m(x, gt=None)
    print(kw, "build+first", t1 - t0, "second", time.time() - t1)
    print(sorted(P), sorted(out), sorted(oe), {k: v.dtype for k, v in oe.items()})
    print("equal", all(torch.equal(oe[k], oe2[k]) for k in oe))
