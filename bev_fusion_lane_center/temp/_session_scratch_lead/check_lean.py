"""Throw-away check (reviewer 2026-11-17): score_preds with lean=True equals lean=False on real frames (TRAIN every-3 dump of e367_b25_s0), and the time per cell."""
import argparse, os, sys, time
import numpy as np

ROOT = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import jtoken_eval as JE
QE = JE.QE

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
a = argparse.Namespace(run="e367_b25_s0", model="model.pt", seed=0, split="train", every=3, drop=0.0, drop_seed=0, live=False, max=0)
r = JE.read_dump(a, JE.RUNS, 0)
quiet = lambda *x, **k: None
fr, tk, va, frames = r["fr"][:N], r["toks"][:N], r["valid"][:N], r["frames"][:N]
orig = QE.score_frames


def run(use_lean):
    def sf(preds, frames_, thr_e=0.5, thr_link=0.5, records=False, lean=True):
        return orig(preds, frames_, thr_e, thr_link, records, lean=use_lean)
    QE.score_frames = sf
    for f in frames:
        f._lane = f._ref = None
    t0 = time.perf_counter()
    out = []
    for cell in ((0.3, 0.2, 0.6, False), (0.1, 0.1, 0.6, False), (0.2, 0.2, 0.5, True)):
        dec = [JE.decode_variant(f, v, cell[0], cell[2], cell[3], cell[1]) for f, v in zip(fr, va)]
        preds = [JE.frame_prediction(t, f, "rec", cell[2]) for t, f in zip(tk, dec)]
        out.append(JE.score_preds(preds, frames, log=quiet))
    return time.perf_counter() - t0, out


tF, full = run(False)
tL, lean = run(True)
QE.score_frames = orig
sys.path.insert(0, os.path.join(ROOT, "temp", "_session_scratch_lead"))
from deepcmp import deep_eq
ok = True
for i, (x, y) in enumerate(zip(full, lean)):
    same_g = deep_eq(x["graph"], y["graph"])
    same_n = deep_eq(x["nodes"], y["nodes"])
    same_j = all(deep_eq(a_["lane"]["junc"], b_["lane"]["junc"]) for a_, b_ in zip(x["rec"]["graph"], y["rec"]["graph"]))
    same_r = all(deep_eq(a_["rec"], b_["rec"]) and deep_eq(a_["topo"], b_["topo"]) and deep_eq(a_["dp"], b_["dp"]) for a_, b_ in zip(x["rec"]["graph"], y["rec"]["graph"]))
    same_c = deep_eq(x["rec"]["node"], y["rec"]["node"]) and deep_eq(x["rec"]["assign"], y["rec"]["assign"])
    print(f"cell {i}: graph {same_g} nodes {same_n} junc {same_j} rec/topo/dp {same_r} node+assign counts {same_c}", flush=True)
    ok &= same_g and same_n and same_j and same_r and same_c
print(f"frames {N}, 3 cells: full {tF:.1f} s, lean {tL:.1f} s, speed-up {tF / tL:.2f}x, all equal: {ok}", flush=True)
sys.exit(0 if ok else 1)
