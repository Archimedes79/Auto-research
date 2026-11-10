"""old vs new evaluation code on real GT frames with old-style (no 1 m head) predictions: per-frame metric counts, reduced numbers and every view's rendered pixels.
    py.sh old_new.py old|new  ->  out_<which>.pkl  (the code of HEAD exported to old/, or the working tree)"""
import os, pickle, sys
import numpy as np
import torch

which = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "old", "bev_fusion") if which == "old" else os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.join(HERE, "old") if which == "old" else os.path.join(ROOT, "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "training"), REPO, os.path.join(REPO, "autoresearch")]
os.environ["MPLBACKEND"] = "Agg"
import evaluation
print(which, "evaluation from", os.path.dirname(evaluation.__path__[0]) if hasattr(evaluation, "__path__") else evaluation.__file__)
from evaluation import bev_gtcheck as GC
from evaluation.bev_task import BEVTask
import evaluation.bev_task as BT
from evalsuite import Prediction, Sample, ViewContext
from evalsuite.contracts import FrameRef
from evalsuite.render.figure import render_page
import losses as L
from lift_pd import Grid

assert os.path.realpath(BT.__file__).startswith(os.path.realpath(ROOT)), (BT.__file__, ROOT)
torch.set_num_threads(2)
SPEC = L.layer_spec(False, True)
GROUPS = [g for g, _ in L.NUSC_GROUPS]
samples = GC.pick(6, "15efbde2d1", 3)
variants = GC.variants(samples, SPEC, GROUPS, seed=2)
task = BEVTask(Grid(), SPEC, GROUPS, refine=False)
task.curves = task.fusion = task.velocity = True
out = {"counts": {}, "reduced": {}, "pixels": {}}
rad = {k: torch.tensor(v, dtype=torch.float32) for k, v in (("x", [5.0, 12.0, 30.0]), ("y", [0.0, 3.0, -4.0]), ("vr", [0.0, 5.0, -3.0]))}
for vname in ("oracle", "noise 0.2", "dropout 50 %", "shuffled"):
    cnt = {m.name + str(i): [] for i, m in enumerate(task.metrics())}
    for i, ((name, s), outs) in enumerate(zip(samples, variants[vname])):
        for o in outs.values():
            o["mass"] = np.where(np.arange(104) < 70, 0.9, 0.1)[None].repeat(120, 0).astype(np.float32)
            o["objvel"] = None
        sample = Sample(ref=FrameRef("gt", i, name), inputs={"radar": rad}, reference=s, display={"camera": None})
        pred = Prediction(outputs={"dt": outs}, taps={}, info={"loss": {"total": 1.0}})
        for j, m in enumerate(task.metrics()):
            cnt[m.name + str(j)].append(m.frame(sample, pred))
        if i < 2:
            for vn, view in task.views().items():
                if vn in ("taps",):
                    continue
                page = view.render(sample, pred, ViewContext())
                out["pixels"][(vname, i, vn)] = (len(page.panels), render_page(page, dpi=50))
    for j, m in enumerate(task.metrics()):
        out["counts"][(vname, m.name, j)] = cnt[m.name + str(j)]
        out["reduced"][(vname, m.name, j)] = m.reduce(cnt[m.name + str(j)])
pickle.dump(out, open(os.path.join(HERE, f"out_{which}.pkl"), "wb"))
print(which, "done:", len(out["reduced"]), "reductions,", sum(len(v) for v in out["reduced"].values()), "numbers,", len(out["pixels"]), "pictures")
