"""scratch (2026-11-12): parameters and MACs of the junction thread's blocks (CPU, batch 1), from the checkpoint of a pair-head run.
usage: wsl bash tools/env/_py14.sh jr/model_cost.py [run]   -> prints a table and writes temp/jr/model_cost.json"""
import json, os, sys
import torch
from torch.utils.flop_counter import FlopCounterMode
ROOT = os.environ["BFLC_ROOT"]
for sub in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import jends_eval as JE
torch.set_num_threads(2)
run = sys.argv[1] if len(sys.argv) > 1 else "e312_p_s0"
m, a = JE.load_model(run, "cpu")
x = torch.randn(1, 39, 200, 200)
npar = lambda mod: sum(p.numel() for p in mod.parameters())
rows = {"pyramid (B's, frozen)": m.jmodel.pyr, "junction maps head": m.jmodel.jm}
for k in dir(m):
    pass
ends = [c for n, c in m.named_children() if n not in ("jmodel",)]
res = {}
with torch.no_grad():
    with FlopCounterMode(display=False) as f_all:
        P, out, oe = m(x, gt=None)
    P = m.jmodel.pyr(x)
    with FlopCounterMode(display=False) as f_pyr:
        P = m.jmodel.pyr(x)
    with FlopCounterMode(display=False) as f_jm:
        jo = m.jmodel.jm(P)
res["total_macs_g"] = f_all.get_total_flops() / 2e9
res["pyr_macs_g"] = f_pyr.get_total_flops() / 2e9
res["jmaps_macs_g"] = f_jm.get_total_flops() / 2e9
res["pair_head_macs_g"] = res["total_macs_g"] - res["pyr_macs_g"] - res["jmaps_macs_g"]
res["params_m"] = dict(pyramid=npar(m.jmodel.pyr) / 1e6, jmaps=npar(m.jmodel.jm) / 1e6, total=npar(m) / 1e6)
res["params_m"]["pair_head"] = res["params_m"]["total"] - res["params_m"]["pyramid"] - res["params_m"]["jmaps"]
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(ROOT, "temp", "jr", "model_cost.json"), "w"), indent=1)
