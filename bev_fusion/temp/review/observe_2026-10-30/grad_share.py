"""Loss weighting of the velocity terms (review: why do both velocity paths stay close to zero?).
1. The Kendall weights at the end of training (log.jsonl) and the gradient norm of every task group at the BEV feature (diag rows).
2. Per loss TERM (hm, off, size, rad, vel, dir of the box head; dense motion; maps ...): norm of d(term)/dS at the trunk output S,
   weighted as in training (Kendall weight of its group), averaged over 12 train frames (batches of 4, EMA weights, eval mode).
3. The composition of the targets: moving / static share of the vehicle cells and of the supervised boxes.
usage: grad_share.py <run>"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from evaluation import bev_probe as ob  # noqa: E402
from train import collate_cached, collate, forward_loss, make_datasets  # noqa: E402

run = sys.argv[1] if len(sys.argv) > 1 else "runs/t1_r4_soa"
rows = [json.loads(l) for l in open(f"{run}/log.jsonl")]
last = [r for r in rows if "s.areas" in r][-1]
print("Kendall s at the end:", {k[2:]: round(v, 2) for k, v in last.items() if k.startswith("s.")},
      "-> weights exp(-s):", {k[2:]: round(float(np.exp(-v)), 2) for k, v in last.items() if k.startswith("s.")})
for r in rows:
    d = r.get("diag")
    if d and r["step"] in (500, 2000, 4000, 7000):
        print(f"step {r['step']}: gradient norm of the groups at the BEV feature", dict(zip(d["groups"], [round(x, 3) for x in d["norm"]])))

cfg, model, loss, step, val_ds, device = ob.load_all(run, "cpu")
train_ds = make_datasets(cfg)[0]
fn = collate_cached if cfg.feature_cache else collate
model.eval()
by = {}
for k, tok in enumerate(train_ds.samples):
    by.setdefault(train_ds.nusc.get("sample", tok)["scene_token"], []).append(k)
idx = [v[len(v) // 2] for v in by.values()]
idx = [idx[int(j)] for j in np.linspace(0, len(idx) - 1, 12).round()]
share = {}
mov_c = stat_c = mov_b = stat_b = 0
for b0 in range(0, 12, 4):
    items = []
    for i in idx[b0:b0 + 4]:
        try:
            items.append(train_ds[i])
        except KeyError:
            pass
    if not items:
        continue
    batch = fn(items)
    out, res, gts = forward_loss(model, loss, batch, 1.0, device, cfg.amp, cfg, train_terms=False)
    S = res["features"]
    w = {g: float(torch.exp(-loss.task.values()[j])) for j, g in enumerate(loss.task.groups)}
    terms = {k: v for k, v in out.items() if torch.is_tensor(v) and v.requires_grad and (k.startswith(("objhead.", "sem.")) or k == "motion")}
    for name, t in terms.items():
        g = torch.autograd.grad(t, S, retain_graph=True, allow_unused=True)[0]
        group = "motion" if name == "motion" else ("objects" if name.startswith("objhead.") else "maps")
        gw = w.get(group, 1.0) if group != "maps" else 1.0
        share.setdefault(name, []).append(float(g.norm()) * gw if g is not None else 0.0)
    for g in gts[0.0]:
        gv = np.asarray(g["velocity"], np.float32)
        veh = np.asarray(g["vehicle"], np.float32) > 0.5
        sp = np.hypot(gv[0], gv[1])
        mov_c += int((veh & (sp > 1.0)).sum())
        stat_c += int((veh & (sp <= 1.0)).sum())
        for bx in g["meta"]["boxes"]:
            if bx["group"] == "vehicle" and bx.get("observed", True):
                v = float(np.hypot(*np.nan_to_num(np.asarray(bx["vel"], np.float64))))
                mov_b += v > 1.0
                stat_b += v <= 1.0
tot = sum(np.mean(v) for v in share.values())
print("\ngradient norm at S per term (x group weight), share of the sum:")
for k, v in sorted(share.items(), key=lambda kv: -np.mean(kv[1])):
    print(f"  {k:28s} {np.mean(v):8.4f}  {100 * np.mean(v) / tot:5.1f} %")
print(f"\nvehicle cells: moving {mov_c} / static {stat_c} -> moving share {100 * mov_c / max(mov_c + stat_c, 1):.0f} %; "
      f"vehicle boxes (observed): moving {mov_b} / static {stat_b} -> moving share {100 * mov_b / max(mov_b + stat_b, 1):.0f} %")
