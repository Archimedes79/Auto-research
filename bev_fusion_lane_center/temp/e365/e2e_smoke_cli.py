"""scratch: dump + score of the smoke head on all 873 VAL frames on the CPU, into a scratch runs folder"""
import os, sys, shutil
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.join(os.environ["BFLC_ROOT"], "evaluation"))
import torch
import jtoken_eval as E
runs = os.path.join(E.SCRATCH, "runs_smoke")
if os.path.exists(runs):
    shutil.rmtree(runs)
os.makedirs(os.path.join(runs, "h"))
shutil.copy(os.path.join(E.ROOT, "runs", "e365_smoke_build", "model.pt"), os.path.join(runs, "h", "model.pt"))
torch.set_num_threads(2)
E.dump(E.parse_args(["dump", "--run", "h", "--seed", "0", "--cpu"]), runs)
E.dump(E.parse_args(["dump", "--run", "h", "--seed", "0", "--cpu", "--drop", "0.17"]), runs)
out = E.score(E.parse_args(["score", "--run", "h", "--seed", "0"]), runs)
E.score(E.parse_args(["score", "--run", "h", "--seed", "0", "--drop", "0.17"]), runs)
print(sorted(os.listdir(os.path.join(runs, "h"))))
c = out["rec"]
print({k: c["metrics"][k] for k in ("topo_F1", "F1@1.0", "lanes_found")}, c["paired_vs_plain_b"]["topo_F1"], c["g1_topo_F1"])
