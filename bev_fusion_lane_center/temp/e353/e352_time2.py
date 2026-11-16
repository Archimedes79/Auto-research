import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch
torch.set_num_threads(2)
ROOT = os.environ["BFLC_ROOT"]
for s in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train"):
    sys.path.insert(0, os.path.join(ROOT, s))
import lane_gt_data as LD
import qpredict as QP
import jends_eval as JEV
S = 0
t0 = time.time()
m = QP.load_model(os.path.join(ROOT, "runs", f"e273_p_s{S}"), "cpu")
jm, _ = JEV.load_model(f"e324_s_s{S}", "cpu")
print("loaded", round(time.time() - t0, 1), flush=True)
t0 = time.time()
ds, dl = LD.loader("val_clean", bs=4, workers=0, shuffle=False, map_mode="exact", pixels=False, strict=True)
print("loader built", round(time.time() - t0, 1), flush=True)
it = iter(dl)
for i in range(2):
    t1 = time.time()
    b = next(it)
    t2 = time.time()
    with torch.no_grad():
        x = b["x"].float()
        P = m.pyr(x)
        t3 = time.time()
        out = m(b, upto="assign", frozen=("pyr", "nodes", "links", "sections", "assign"), gt=False)
        t4 = time.time()
        jo = jm(x, gt=None)
        t5 = time.time()
    print(f"batch {i}: load {t2 - t1:.1f} s, B pyramid alone {t3 - t2:.1f} s, B full {t4 - t3:.1f} s, junction model {t5 - t4:.1f} s", flush=True)
    t6 = time.time()
    preds = QP.predict_from_out(out, b)
    print(f"  predict_from_out {time.time() - t6:.1f} s", flush=True)
