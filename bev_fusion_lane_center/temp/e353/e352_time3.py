import os, sys, time, faulthandler
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch
torch.set_num_threads(2)
ROOT = os.environ["BFLC_ROOT"]
for s in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train"):
    sys.path.insert(0, os.path.join(ROOT, s))
import lane_gt_data as LD
import qpredict as QP
fh = open(os.path.join(ROOT, "temp/e353/time3_stacks.log"), "w")
faulthandler.dump_traceback_later(40, repeat=True, file=fh)
m = QP.load_model(os.path.join(ROOT, "runs", "e273_p_s0"), "cpu")
ds, dl = LD.loader("val_clean", bs=4, workers=0, shuffle=False, map_mode="exact", pixels=False, strict=True)
it = iter(dl)
t0 = time.time(); b = next(it); print("batch loaded", round(time.time() - t0, 1), flush=True)
t0 = time.time()
with torch.no_grad():
    P = m.pyr(b["x"].float())
print("pyramid", round(time.time() - t0, 1), flush=True)
t0 = time.time()
preds = QP.predict(m, b)
print("predict", round(time.time() - t0, 1), flush=True)
