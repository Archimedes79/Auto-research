import os, sys, tempfile
sys.path.insert(0, "evaluation"); sys.path.insert(0, ".")
import numpy as np
import evaluation.bev_selftest as T
from evaluation import bev_suite
T.init_synthetic()
X, Y = T._centres()
inter = {"fovj": (X > 20) & (X < 30) & (np.abs(Y) < 6), "fovjb": (X > 19.5) & (X < 30.5) & (np.abs(Y) < 6.5)}
with tempfile.TemporaryDirectory() as d:
    a, plain = T.make_suite(d), T.make_suite(d)
    a.task.junction_free, a.task._junction = True, (lambda g: inter)
    sa, sp = a.quant.run(name="ja", progress=0), plain.quant.run(name="jp", progress=0)
    fa, fp = bev_suite.to_legacy(a.quant.table(sa)), bev_suite.to_legacy(plain.quant.table(sp))
    std = {k: v for k, v in fa.items() if not k.startswith(("seg_fovj/", "seg_fovjb/"))}
    print(len(std), len(fp), set(std) ^ set(fp))
    bad = [(k, std[k], fp[k]) for k in std if k in fp and std[k] != fp[k]]
    print(len(bad), bad[:5])
