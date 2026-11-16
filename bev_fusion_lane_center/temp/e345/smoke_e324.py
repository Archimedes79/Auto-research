import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.join(os.environ["BFLC_ROOT"], "evaluation"))
import jprune_probe as P
torch.set_num_threads(2)
out = os.path.join(os.environ["BFLC_ROOT"], "temp/e345/e324_smoke")
os.makedirs(out, exist_ok=True)
rd = os.path.join(os.environ["BFLC_ROOT"], "temp/e345/e324_smoke_runs")
os.makedirs(os.path.join(rd, "e324_s_s0"), exist_ok=True)
if not os.path.lexists(os.path.join(rd, "e324_s_s0", "model.pt")):
    os.symlink(os.path.join(os.environ["BFLC_ROOT"], "runs/e324_s_s0/model.pt"), os.path.join(rd, "e324_s_s0", "model.pt"))
sw = P.Sweep("e324_s_s0", "cpu", out, rd, 0, 16, workers=0, val_stride=120)
print("attach", sw.rec["attach"], "frames", len(sw.rows), "units", len(sw.units))
sw.nulls(False)
sw.plumbing()
sw.do("p001", {"pyr.d0": 0.25}); sw.do("p002", {"je.paint": 1.0}); sw.do("p003", {"je.blocks.0.att.h1": 1.0}); sw.do("p004", {"je.scale.1": 0.5})
sw.dump_point(3, {"pyr.e1": 0.25, "je.paint": 1.0})
r = sw.rec["passes"]["null"]
print("kp sums", np.sum(r["kp"], 0), "ap", r["pair"])
import pickle
fr = pickle.load(open(os.path.join(rd, "e345_p003_e324_s_s0", "ends_val.pkl"), "rb"))
print(len(fr), sorted(fr[0]))
