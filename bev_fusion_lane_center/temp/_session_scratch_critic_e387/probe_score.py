import os, sys, time, argparse
R = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(R, "evaluation"))
import torch
torch.set_num_threads(1)
import jtoken_eval as JE
t0 = time.time()
run = sys.argv[1]
a = argparse.Namespace(run=run, seed=0, split="val", every=1, drop=0.0, drop_seed=0, max=0, live=False)
r = JE.read_dump(a, JE.RUNS, 0)
print("read_dump ok", len(r["fr"]), "frames", round(time.time()-t0), "s", flush=True)
JE.check_record(r["fr"], r["valid"])
print("check_record ok", flush=True)
res = JE.score_cell(r, (0.1, 0.1, 0.6, False), r["frames"], log=lambda *x, **k: None)
print({k: round(float(res["graph"][k]), 4) for k in JE.KEYS}, round(time.time()-t0), "s")
