import argparse, cProfile, pstats, sys, time, os, pickle
import numpy as np
ROOT = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import jtoken_eval as JE

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
a = argparse.Namespace(run="e367_b25_s0", model="model.pt", seed=0, split="train", every=3, drop=0.0, drop_seed=0, live=False, max=0)
t0 = time.time()
r = JE.read_dump(a, JE.RUNS, 0)
print("read_dump", round(time.time() - t0, 1), "s", len(r["fr"]), "frames", flush=True)
t0 = time.time()
JE.check_record(r["fr"], r["valid"])
print("check_record", round(time.time() - t0, 2), "s", flush=True)
fr, tk, va, frames = r["fr"][:N], r["toks"][:N], r["valid"][:N], r["frames"][:N]
cell = (0.3, 0.2, 0.6, False)
quiet = lambda *x, **k: None

t0 = time.time()
dec = [JE.decode_variant(f, v, cell[0], cell[2], cell[3], cell[1]) for f, v in zip(fr, va)]
t_dec = time.time() - t0
t0 = time.time()
preds = [JE.frame_prediction(t, f, "rec", cell[2]) for t, f in zip(tk, dec)]
t_pred = time.time() - t0
t0 = time.time()
res = JE.score_preds(preds, frames, log=quiet)
t_score = time.time() - t0
print(f"N={N}: decode {t_dec:.3f}s  frame_prediction {t_pred:.3f}s  score_preds {t_score:.2f}s  ({t_score / N * 1000:.0f} ms/frame)", flush=True)
t0 = time.time()
res2 = JE.score_preds(preds, frames, log=quiet)
print(f"second pass score_preds {time.time() - t0:.2f}s", flush=True)

pr = cProfile.Profile()
pr.enable()
JE.score_preds(preds, frames, log=quiet)
pr.disable()
st = pstats.Stats(pr)
st.sort_stats("cumulative").print_stats(32)
