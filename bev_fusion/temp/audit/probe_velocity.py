"""One-off audit (2026-10-26): is there velocity information in the trunk features of a trained run, i.e. is the velocity error of
the object head (about that of 'standing still') a question of the loss weight or of the information? Everything of the run is
frozen; a fresh small head (3x3 conv + 1x1 conv) is trained on the trunk features at dt 0 to predict the GT velocity at the GT
centre cells (L1, like object_head.head_loss), on train samples; then the L1 on val centres is compared with (a) predicting 0,
(b) the run's own velocity branch. Usage: python temp/audit/probe_velocity.py runs/t1_r3_combined --steps 400
"""
import argparse
import os
import sys

import torch
import torch.nn as nn

sys.argv, _argv = [sys.argv[0], next(a for a in sys.argv[1:] if not a.startswith("-"))], sys.argv
ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--n_val", type=int, default=300)
a = ap.parse_args(_argv[1:])
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "evaluation"))
sys.argv = [sys.argv[0], a.run]
import evaluate_run as E  # noqa: E402
from losses import head_box  # noqa: E402
from object_head import _gather, build_targets  # noqa: E402


def targets(gts0, grid):
    boxes = [[hb for bx in g["meta"]["boxes"] if bx.get("observed", True) and (hb := head_box(bx)) is not None] for g in gts0]
    return build_targets(boxes, grid, E.NUSC_GROUPS, 128)


def main():
    device = "cuda"
    os.chdir(E.ROOT)
    cfg, model, loss, step, ckpt = E.load_run(a.run, device)
    cfg.batch_size, cfg.workers, cfg.seq_len = 8, 4, 1
    if not cfg.radar_cache and os.path.isdir("data/cache/radar"):
        cfg.radar_cache = "data/cache/radar"
    train_ds, val_ds = E.make_datasets(cfg)
    for p in model.parameters():
        p.requires_grad_(False)
    c = model.c_trunk
    probe = nn.Sequential(nn.Conv2d(c, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 2, 1)).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    tl = E.make_loader(train_ds, cfg, True)
    it, i = iter(tl), 0
    while i < a.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(tl)
            continue
        with torch.no_grad():
            _, res, gts = E.forward_loss(model, loss, batch, 1.0, device, cfg.amp, cfg, train_terms=False)
        t = targets(gts[0.0], model.grid)
        S = res["features"].detach().float()
        pred = _gather(probe(S), t["ind"].to(device))
        m = t["mask"].to(device).unsqueeze(-1)
        l1 = ((pred - t["vel"].to(device)).abs() * m).sum() / (2 * m.sum()).clamp(min=1)
        opt.zero_grad()
        l1.backward()
        opt.step()
        i += 1
        if i % 100 == 0:
            print(f"probe step {i}: train L1 {float(l1):.3f}", flush=True)
    vl = E.make_loader(E.val_subset(val_ds, a.n_val), cfg, False)
    acc = {k: [0.0, 0.0, 0.0, 0.0] for k in ("zero", "run_head", "probe")}      # sum |err| per component pair, count, for vehicles / all
    sums = {k: {"veh": [0.0, 0], "all": [0.0, 0]} for k in acc}
    with torch.no_grad():
        for batch in vl:
            _, res, gts = E.forward_loss(model, loss, batch, 1.0, device, cfg.amp, cfg, train_terms=False)
            t = targets(gts[0.0], model.grid)
            S = res["features"].float()
            preds = {"zero": torch.zeros_like(t["vel"]).to(device),
                     "run_head": _gather(res["out"][0.0]["obj"]["vel"].float(), t["ind"].to(device)),
                     "probe": _gather(probe(S), t["ind"].to(device))}
            m = t["mask"].to(device)
            veh = m * (t["group"].to(device) == 0).float()
            for k, p in preds.items():
                e = (p - t["vel"].to(device)).norm(dim=-1)                          # speed error per object (m/s)
                sums[k]["all"][0] += float((e * m).sum()); sums[k]["all"][1] += int(m.sum())
                sums[k]["veh"][0] += float((e * veh).sum()); sums[k]["veh"][1] += int(veh.sum())
    for k, v in sums.items():
        print(f"{k:9s} velocity error (m/s, Euclidean at the GT centre cells): vehicles {v['veh'][0] / max(v['veh'][1], 1):.2f} "
              f"(n {v['veh'][1]}), all objects {v['all'][0] / max(v['all'][1], 1):.2f}")


main()
