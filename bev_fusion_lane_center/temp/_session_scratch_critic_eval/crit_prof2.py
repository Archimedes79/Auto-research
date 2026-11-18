import argparse, sys, time, os, pickle
import numpy as np
ROOT = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import jtoken_eval as JE
QE = JE.QE

a = argparse.Namespace(run="e367_b25_s0", model="model.pt", seed=0, split="train", every=3, drop=0.0, drop_seed=0, live=False, max=0)
r = JE.read_dump(a, JE.RUNS, 0)
cell = (0.3, 0.2, 0.6, False)
quiet = lambda *x, **k: None
N = 30
fr, tk, va, frames = r["fr"][:N], r["toks"][:N], r["valid"][:N], r["frames"][:N]
dec = [JE.decode_variant(f, v, cell[0], cell[2], cell[3], cell[1]) for f, v in zip(fr, va)]
preds = [JE.frame_prediction(t, f, "rec", cell[2]) for t, f in zip(tk, dec)]
# A: baseline (ref dropped after each frame)
t0 = time.time(); JE.score_preds(preds, frames, log=quiet); tA = time.time() - t0
# B: refs cached
t0 = time.time()
with JE.JT.lattice(JE.LATTICE):
    for f in frames: f.ref()
tL = time.time() - t0
t0 = time.time(); resB = JE.score_preds(preds, frames, log=quiet); tB = time.time() - t0
print(f"A baseline {tA:.2f}s  load refs once {tL:.2f}s  B with refs cached {tB:.2f}s  -> {100*(1-tB/tA):.0f}% less per further cell", flush=True)
sz = sum(len(pickle.dumps(f._lane)) + len(pickle.dumps(f._ref.__dict__)) for f in frames) / N
print("avg pickled size lane+ref per frame (bytes)", int(sz), flush=True)
# full 291 frames, one cell, records
fr, tk, va, frames = r["fr"], r["toks"], r["valid"], r["frames"]
dec = [JE.decode_variant(f, v, cell[0], cell[2], cell[3], cell[1]) for f, v in zip(fr, va)]
preds = [JE.frame_prediction(t, f, "rec", cell[2]) for t, f in zip(tk, dec)]
t0 = time.time(); res = JE.score_preds(preds, frames, log=quiet); print("full 291 frames one cell", round(time.time() - t0, 1), "s", flush=True)
dec2 = [JE.decode_variant(f, v, 0.2, 0.5, True, 0.2) for f, v in zip(fr, va)]
preds2 = [JE.frame_prediction(t, f, "rec", 0.5) for t, f in zip(tk, dec2)]
res2 = JE.score_preds(preds2, frames, log=quiet)
sc = np.array([f.scene for f in frames]); print("scenes", len(set(sc.tolist())))
t0 = time.time(); p1 = QE.paired(res["rec"], res2["rec"], frames); print("QE.paired", round(time.time() - t0, 1), "s", flush=True)
t0 = time.time(); p2 = JE.paired_junction(res["rec"], res2["rec"], frames); print("paired_junction", round(time.time() - t0, 1), "s", flush=True)
t0 = time.time(); QE.SB.paired([c["rec"] for c in res["rec"]["graph"]], [c["rec"] for c in res2["rec"]["graph"]], sc); print("  SB.paired", round(time.time() - t0, 1), flush=True)
t0 = time.time(); QE.TM.paired([c["topo"] for c in res["rec"]["graph"]], [c["topo"] for c in res2["rec"]["graph"]], sc); print("  TM.paired", round(time.time() - t0, 1), flush=True)
print({k: (round(v["diff"], 4), round(v["lo"], 4), round(v["hi"], 4)) for k, v in list(p1.items())[:3]})
print(p2)
