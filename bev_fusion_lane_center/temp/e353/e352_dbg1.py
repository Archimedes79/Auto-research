import os, sys, torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(2)
ROOT = os.environ["BFLC_ROOT"]
for s in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train"):
    sys.path.insert(0, os.path.join(ROOT, s))
import jprune_b as PB, jprune_cost as PC, jprune_probe as PP, jends_eval as JEV
jm, _ = JEV.load_model("e324_s_s0", "cpu")
calib = PP.load_calib(PP.rootpath("temp/e345/e324_s_s0/calib.pt"), "cpu")
cost = PC.Cost(jm)
a = cost.saved(PB.point_edits(jm, calib, "t1")); b = cost.saved(PB.point_edits(jm, calib, "r1d1"))
print({k: (a[k], b[k]) for k in a if a[k] != b[k]})
