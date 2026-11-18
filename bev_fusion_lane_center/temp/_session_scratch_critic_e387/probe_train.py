import os, sys, time, argparse
R = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(R, "evaluation"))
import torch
torch.set_num_threads(1)
import jtoken_eval as JE
t0 = time.time()
for run in sys.argv[1:]:
    a = argparse.Namespace(run=run, seed=int(run[-1]), split="train", every=3, drop=0.0, drop_seed=0, max=0, live=False)
    r = JE.read_dump(a, JE.RUNS, 0)
    print(run, "TRAIN every3 read_dump ok", len(r["fr"]), "frames", round(time.time()-t0), "s", flush=True)
    JE.check_record(r["fr"], r["valid"])
    cell = (0.3, 0.2, 0.6, False)
    n = 40
    fr = [JE.decode_variant(f, v, cell[0], cell[2], cell[3], cell[1]) for f, v in zip(r["fr"][:n], r["valid"][:n])]
    res = JE.score_preds([JE.frame_prediction(t, f, "rec", cell[2]) for t, f in zip(r["toks"][:n], fr)], r["frames"][:n], log=lambda *x, **k: None)
    print(run, "D115 on first", n, {k: round(float(res["graph"][k]), 4) for k in ("topo_F1", "lanes_found", "junction_F1", "dup_frac")}, round(time.time()-t0), "s", flush=True)
