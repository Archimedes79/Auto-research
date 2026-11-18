import argparse, sys, time, os
import numpy as np
ROOT = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import jtoken_eval as JE
QE = JE.QE
N = 30
a = argparse.Namespace(run="e367_b25_s0", model="model.pt", seed=0, split="train", every=3, drop=0.0, drop_seed=0, live=False, max=0)
r = JE.read_dump(a, JE.RUNS, 0)
cell = (0.3, 0.2, 0.6, False)
quiet = lambda *x, **k: None
fr, tk, va, frames = r["fr"][:N], r["toks"][:N], r["valid"][:N], r["frames"][:N]
dec = [JE.decode_variant(f, v, cell[0], cell[2], cell[3], cell[1]) for f, v in zip(fr, va)]
preds = [JE.frame_prediction(t, f, "rec", cell[2]) for t, f in zip(tk, dec)]
orig_tier = QE.tier_counts
def run(tier_off, cache):
    QE.tier_counts = (lambda raw, ref: None) if tier_off else orig_tier
    for f in frames: f._lane = f._ref = None
    t0 = time.perf_counter()
    if cache:
        with JE.JT.lattice(JE.LATTICE):
            for f in frames: f.ref()
    res = JE.score_preds(preds, frames, log=quiet)   # includes ref load once when cache (counted: t0 before)
    return time.perf_counter() - t0, res
run(False, False)  # warm
out = {}
base_res = None
for rep in range(3):
    for name, (to, ca) in dict(base=(False, False), tier_off=(True, False), ref_cached_once=(False, True), both=(True, True)).items():
        dt, res = run(to, ca)
        out.setdefault(name, []).append(dt)
        if name == "base": base_res = res
        else:
            same = all(abs(res["graph"][k] - base_res["graph"][k]) < 1e-15 for k in res["graph"])
            out.setdefault(name + "_same", []).append(same)
for k, v in out.items():
    print(k, [round(x, 2) if not isinstance(x, bool) else x for x in v])
