"""E364 arm n / the fixture: the end-to-end read-out of E324 s (`--attach --snap --snap-nogeom --thr 0.1`) scored by the UNCHANGED `jends_eval.score`, with the long-link cover radius `qends_eval.COVER_POS` set to -1.0 IN PROCESS (arm n, tag e364n) or left at 4.0 (`--fixture`, tag e364f: must reproduce the stored PRIMARY card).
usage (problem folder, CPU): wsl bash tools/env/_py14.sh score_nocover.py --run e324_s_s0 --seed 0 [--fixture]"""
import argparse
import os
import sys
from types import SimpleNamespace as NS

ROOT = os.environ["BFLC_ROOT"]
for sub in ("evaluation", "training", "training/grid", "training/data", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import jends_eval as JE  # noqa: E402
import qends_eval as QE  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--fixture", action="store_true")
a = ap.parse_args()
assert QE.COVER_POS == 4.0, QE.COVER_POS
if not a.fixture:
    QE.COVER_POS = -1.0                                                                                    # in process: `cover_pos()` reads the module global at call time; -1 (not 0) so that even an exact coincidence of ends does not drop a link
    assert QE.cover_pos() == -1.0 and QE.cover_pos(None) == -1.0
tag = "e364f" if a.fixture else "e364n"
print(f"score {a.run} seed {a.seed}: COVER_POS = {QE.COVER_POS}, tag {tag}", flush=True)
JE.score(NS(run=a.run, seed=a.seed, thr=0.1, tag=tag, snap=True, attach=True, snap_nogeom=True))
