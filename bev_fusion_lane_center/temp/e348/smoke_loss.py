"""E348 CPU smoke (the trainer needs CUDA, so the exact `jends_train.py` command cannot run on the CPU): the model, the loader and the loss of the exact E348 command, ONE real TRAIN batch, forward + loss + backward on the CPU.
Checks: the pair head is built with 150 inputs (evidence without the two paint columns), the maps head has no 0.5 m branch, the attachment exists, the loss is finite and has the geometry / attachment terms, the gradient reaches the pair head,
the attachment and the maps head (e2e), and not the frozen pyramid and B's node head.
usage: wsl bash tools/env/_py14.sh temp/e348/smoke_loss.py [--seed 0] [--bs 2]   (CUDA hidden; scratch)"""
import argparse
import os
import sys

import torch

ROOT = os.environ.get("BFLC_ROOT") or os.getcwd()
for sub in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import jends as JE  # noqa: E402
import jends_train as JT  # noqa: E402
import jmaps as JM  # noqa: E402
import lane_gt_data as LD  # noqa: E402
import trainer as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--bs", type=int, default=2)
a = ap.parse_args()
torch.set_num_threads(2)
os.environ["BFLC_SIG"] = "1.0"
T.seed_all(a.seed)
LD.QData.DEFAULT_JUNCTION = True
LD.QData.DEFAULT_PSCALE = True                                                                           # --scales
LD.QData.DEFAULT_PPTS = True                                                                             # --geom-w 0.5
mp = os.path.join(ROOT, "runs", f"e332_ma_s{a.seed}", "model.pt")
ck = torch.load(mp, map_location="cpu")
model = JT.JEndModel(ck["args"], order=True, evidence=True, scales=True, e2e=True, ntok=False, unc=False, attach=True)
model.jmodel.load_state_dict(ck["state"])
base = ck["args"].get("base") or os.path.join(ROOT, "runs", f"e273_p_s{a.seed}", "model.pt")
bs_ = torch.load(base, map_location="cpu")["state"]
model.nodes.load_state_dict({k[6:]: v for k, v in bs_.items() if k.startswith("nodes.")})
print("maps ctx:", ck["args"].get("ctx"), "| base:", base)
assert "nopaint" in ck["args"].get("ctx", ""), "the maps checkpoint is not the no-paint head"
assert not hasattr(model.jmodel.jm, "m05") or model.jmodel.jm.m05 is None, "the maps head still has the 0.5 m branch"
nin = model.je.pair[0].in_features
print("pair MLP inputs:", nin, "(E312: 152, no paint columns: 150)")
assert nin == 150, nin
assert model.je.attach is not None, "no attachment"
for p in model.parameters():
    p.requires_grad_(False)
ps = list(model.je.parameters()) + list(model.jmodel.jm.parameters())
for p in ps:
    p.requires_grad_(True)
ds, dl = LD.loader("train", bs=a.bs, workers=0, shuffle=True, map_mode="train", pixels=False)
b = next(iter(dl))
x = b["x"].float()
gt = {k: b[k] for k in ("ent", "ext", "ent_m", "ext_m")}
model.train()
_, jout, out = model(x, gt=gt)
loss, info = JE.jends_loss(out, b, cross_w=1.0, driven_w=0.0, geom_w=0.5)
lm, im = JM.jmaps_loss(jout, b)
loss = loss + 1.0 * lm
print("loss", float(loss), {k: round(float(v), 4) for k, v in info.items() if k in ("geom_n", "pos_pairs", "geom_end_d", "geom_shape", "pair", "bce")})
assert torch.isfinite(loss), "the loss is not finite"
assert info.get("geom_n", 0) > 0 or info.get("pos_pairs", 0) == 0, "no pair has a GT polyline although there are positives: the geometry loss does not see its data"
loss.backward()


def gnorm(m):
    return sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)


g_je, g_att, g_jm = gnorm(model.je), gnorm(model.je.attach), gnorm(model.jmodel.jm)
g_pyr, g_nodes = gnorm(model.jmodel.pyr), gnorm(model.nodes)
print(f"gradient sums: pair head {g_je:.3g} (attachment {g_att:.3g}), maps head {g_jm:.3g}, frozen pyramid {g_pyr:.3g}, B's node head {g_nodes:.3g}")
assert g_je > 0 and g_att > 0 and g_jm > 0, "a module that must train got no gradient"
assert g_pyr == 0 and g_nodes == 0, "a frozen module got a gradient"
print("E348 smoke ok")
