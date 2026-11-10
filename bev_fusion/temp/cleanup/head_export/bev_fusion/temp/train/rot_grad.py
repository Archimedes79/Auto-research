"""One-off: does the object head's rot branch train? (1) change of its weights between the run's checkpoints next to the
size branch; (2) gradient norm of the rot loss term on its parameters for one train batch."""
import glob
import sys

import torch

sys.path.insert(0, "training")
sys.path.insert(0, ".")
from evaluation.bev_probe import load_run  # noqa: E402
from train import forward_loss, make_datasets, make_loader, val_subset  # noqa: E402

run = sys.argv[1]
cks = sorted(glob.glob(f"{run}/ckpt_*.pt"))
sd = [torch.load(c, map_location="cpu", weights_only=False)["state"] for c in cks]
for br in ("rot", "size", "off", "hm"):
    keys = [k for k in sd[0] if f"obj_head.head.{br}." in k and "weight" in k]
    for k in keys:
        a = [s[k].float() for s in sd]
        print(br, k.split("head.")[-1], "norms", [round(float(x.norm()), 4) for x in a],
              "change", [round(float((a[i + 1] - a[i]).norm()), 5) for i in range(len(a) - 1)])
print("checkpoints", [c.split("/")[-1] for c in cks])
cfg, model, loss, step, ckpt = load_run(run, "cuda")
model.train()
cfg.workers = 1
train_ds, _ = make_datasets(cfg)
batch = next(iter(make_loader(val_subset(train_ds, 4), cfg, False)))
out, res, gts = forward_loss(model, loss, batch, 1.0, "cuda", cfg.amp, cfg, train_terms=False)
for term in ("objhead.rot@+0.0", "objhead.size@+0.0", "objhead.hm@+0.0"):
    model.zero_grad(set_to_none=True)
    out[term].backward(retain_graph=True)
    g = {br: sum(float(p.grad.norm() ** 2) for p in getattr(model.obj_head.head, br).parameters() if p.grad is not None) ** 0.5
         for br in ("rot", "size", "hm")}
    print(term, float(out[term]), "grad norm per branch", {k: f"{v:.2e}" for k, v in g.items()})
