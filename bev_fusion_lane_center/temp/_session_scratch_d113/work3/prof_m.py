import os, sys, time, tempfile, cProfile, pstats
R = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "training/data/prep", "training/train", "evaluation"):
    sys.path.insert(0, os.path.join(R, sub))
import jmap_train as T
with tempfile.TemporaryDirectory() as tmp:
    a = T.parse_args(["--name", "m", "--smoke", "--bs", "4", "--seed", "0"])
    pr = cProfile.Profile()
    pr.enable()
    T.train(a, runs=tmp, ckpt_every=500, smoke_k=512)
    pr.disable()
    pstats.Stats(pr).sort_stats("cumulative").print_stats(14)
