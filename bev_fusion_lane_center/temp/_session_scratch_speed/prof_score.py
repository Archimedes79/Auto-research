"""CPU profile of the token scorer (scratch): one cell on the first N frames of the VAL / TRAIN dump of a run; serial vs chunked-in-processes equality and timing.
usage: python prof_score.py RUN SEED SPLIT EVERY N"""
import cProfile
import os
import pstats
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.join(os.environ.get("BFLC_ROOT", "."), "evaluation"))
import jtoken_eval as JE  # noqa: E402

run, seed, split, every, n = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
a = JE.parse_args(["score", "--run", run, "--seed", str(seed), "--split", split, "--every", str(every)])
t0 = time.time()
r = JE.read_dump(a, JE.RUNS, None)
print("read_dump", round(time.time() - t0, 1), "s", len(r["frames"]), "frames", flush=True)
r = dict(r, toks=r["toks"][:n], fr=r["fr"][:n], valid=r["valid"][:n], frames=r["frames"][:n])
cell = JE.D115
pr = cProfile.Profile()
t0 = time.time()
pr.enable()
res = JE.score_cell(r, cell, r["frames"], log=lambda *x, **k: None)
pr.disable()
print("score_cell", n, "frames", round(time.time() - t0, 1), "s", {k: round(v, 4) for k, v in res["graph"].items() if k in JE.KEYS}, flush=True)
pstats.Stats(pr).sort_stats("cumulative").print_stats(28)
