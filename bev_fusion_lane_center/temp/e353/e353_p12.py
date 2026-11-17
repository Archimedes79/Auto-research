import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
torch.set_num_threads(2)
ROOT = os.environ["BFLC_ROOT"]
for s in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train"):
    sys.path.insert(0, os.path.join(ROOT, s))
import jnarrow as JN
import jprune_probe as PP
import jprune_cost as PC
SETS = {"n1": {"jm.m05.hid": 0.75}, "n2_dropped": {"jm.m05.hid": 0.75, "pyr.s05+m05": 0.75}, "P1": {"pyr.e2": 0.5, "pyr.s05+m05": 0.25}, "P2": {"pyr.e2": 0.5, "pyr.s05+m05": 0.75, "pyr.d0": 0.25}}
for S in (0, 1):
    full, _ = JN.JEV.load_model(f"e324_s_s{S}", "cpu")
    cost = PC.Cost(full)
    by = {u.name: u for u in PP.build_units(full)}
    for name, spec in SETS.items():
        sv = cost.saved([e for n, f in spec.items() for e in PP.count_edits(by[n], f, cost.C)])
        print(f"seed {S} {name:11s} {spec}: saved {sv['macs'] / 1e9:.4f} GMACs (pyramid {sv['macs_pyr'] / 1e9:.4f}, maps {sv['macs_maps'] / 1e9:.4f}, pair {sv['macs_pair'] / 1e9:.4f}), {sv['params']} params; of {cost.all['macs'] / 1e9:.3f} GMACs / {cost.all['params'] / 1e6:.3f} M (Cost table, node head and attachment not counted)")
