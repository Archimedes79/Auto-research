import os, sys
import torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("training", "training/grid", "training/data", "training/data/prep", "training/train", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
torch.set_num_threads(1)
import jends as JE
import jends_train as JT
import lane_gt_data as LD
LD.QData.DEFAULT_JUNCTION = True
ck = torch.load(os.path.join(ROOT, "runs", "e286_j1_s0", "model.pt"), map_location="cpu")
ds, dl = LD.loader("train", bs=2, workers=0, shuffle=True, map_mode="train", pixels=False)
it = iter(dl); batches = [next(it), next(it)]
m = JT.JEndModel(ck["args"], order=True, evidence=True, ntok=True)
m.jmodel.load_state_dict(ck["state"])
bs_ = torch.load(os.path.join(ROOT, "runs", "e273_p_s0", "model.pt"), map_location="cpu")["state"]
res = m.nodes.load_state_dict({k[6:]: v for k, v in bs_.items() if k.startswith("nodes.")})
print("node head load:", res)
for p in m.parameters(): p.requires_grad_(False)
ps = list(m.je.parameters())
for p in ps: p.requires_grad_(True)
opt = torch.optim.Adam(ps, lr=1e-3)
for b in batches:
    gt = {k: b[k] for k in ("ent", "ext", "ent_m", "ext_m")}
    P, jout, out = m(b["x"].float(), gt=gt)
    loss, info = JE.jends_loss(out, b)
    assert torch.isfinite(loss)
    opt.zero_grad(); loss.backward(); opt.step()
print("ntok smoke ok, loss %.3f, bce %.3f" % (float(loss), info["bce"]))
