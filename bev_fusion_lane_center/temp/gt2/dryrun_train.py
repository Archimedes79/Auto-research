"""CPU dry run of jmaps_train / jends_train (T.fit forced to the CPU, 20 steps, batch 2, no workers)."""
import functools
import os
import sys

R = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "training/data/prep", "training/train", "evaluation"):
    sys.path.insert(0, os.path.join(R, sub))
import trainer as T

orig = T.fit
T.fit = functools.partial(orig, device="cpu", amp="none")
which = sys.argv[1]
if which == "jmaps":
    import jmaps_train as JT
    sys.argv = ["jmaps_train.py", "--name", "dry_jmaps_s0", "--seed", "0", "--ctx", sys.argv[2], "--smoke", "--workers", "0", "--bs", "2"] + sys.argv[3:]
else:
    import jends_train as JT
    sys.argv = ["jends_train.py", "--name", "dry_jends_s0", "--seed", "0", "--maps", os.path.join(R, "runs", "dry_jmaps_s0", "model.pt"), "--smoke", "--workers", "0", "--bs", "2"] + sys.argv[2:]
JT.main()
print("DRY RUN OK", which)
