import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["BFLC_SIG"] = "1.0"
import torch
torch.set_num_threads(2)
ROOT = os.environ["BFLC_ROOT"]
for s in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train"):
    sys.path.insert(0, os.path.join(ROOT, s))
import trainer as T
import jnarrow_train as JTR
out = os.path.join(ROOT, "temp", "e353", "runs_smoke")
os.makedirs(out, exist_ok=True)
orig = T.Run
T.Run = lambda name: orig(name, root=out)
t0 = time.time()
JTR.main(sys.argv[1:])
print("total", time.time() - t0)
