import os, sys, torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(2)
ROOT = os.environ["BFLC_ROOT"]
for s in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train"):
    sys.path.insert(0, os.path.join(ROOT, s))
import jprune_b as PB
jm, args, qg = PB.load_models(1, "cpu")
print("seed 1: B's 57 pyramid tensors equal the junction system's:", PB.same_pyramid(qg, jm) == 57)
import jprune_probe as PP
calib = PP.load_calib(PP.rootpath("temp/e345/e324_s_s1/calib.pt"), "cpu")
for n in ("t1", "t2", "r1d0"):
    print(n, {e.tap: len(e.idx) for e in PB.point_edits(jm, calib, n)})
