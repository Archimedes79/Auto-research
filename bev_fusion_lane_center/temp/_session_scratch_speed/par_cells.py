"""scratch: does scoring several decode cells of one dump in a fork pool give the same numbers as serially, and how much faster?
usage: python par_cells.py RUN SEED SPLIT EVERY NCELLS JOBS"""
import multiprocessing as mp
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.join(os.environ["BFLC_ROOT"], "evaluation"))
import jtoken_eval as JE  # noqa: E402

run, seed, split, every, ncell, jobs = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
a = JE.parse_args(["sweep", "--run", run, "--seed", str(seed), "--split", split, "--every", str(every), "--grid", "perclass"])
R = JE.read_dump(a, JE.RUNS, None)
CELLS = JE.SWEEP_GRIDS["perclass"][:ncell]


def one(cell):
    lt, ltl, kt, gate = cell
    fr = [JE.decode_variant(f, v, lt, kt, gate, ltl) for f, v in zip(R["fr"], R["valid"])]
    res = JE.score_preds([JE.frame_prediction(t, f, "rec", kt) for t, f in zip(R["toks"], fr)], R["frames"], log=lambda *x, **k: None)
    return {k: res["graph"][k] for k in JE.SWEEP_KEYS}


t0 = time.time()
ser = [one(c) for c in CELLS]
t_ser = time.time() - t0
t0 = time.time()
with mp.get_context("fork").Pool(jobs) as pool:
    par = pool.map(one, CELLS, chunksize=1)
t_par = time.time() - t0
print(f"cells {ncell}  serial {t_ser:.1f} s  pool({jobs}) {t_par:.1f} s  speedup {t_ser / t_par:.2f}  identical {ser == par}", flush=True)
