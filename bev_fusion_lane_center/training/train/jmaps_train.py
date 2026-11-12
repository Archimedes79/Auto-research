"""Trainer of the junction maps head (E286, D82): entry / exit heat + heading, junction zone, painted stop lines / crossings, on the pixel stage of a baseline run; start it through `autoresearch/tools/run.py`.

    wsl bash tools/env/_py14.sh jmaps_train.py --name e286_j0_s0 --seed 0 --base runs/e273_p_s0/model.pt [--ctx none|dil|attn] [--joint] [--steps 3000] [--smoke]

`--joint` also trains the pixel stage (the pyramid, with its own pixel loss kept so that B's maps do not drift; the nodes / links of B no longer match such a pixel stage, so a joint run is judged on the
maps only); without it the pyramid is frozen and the junction head reads B's own features. Frames: TRAIN through `lane_gt_data.QData(junction=True)` (the paint sidecars of
`data/prep/junction_paint.py` must exist for TRAIN and VAL); TEST never. Output: `runs/<name>/model.pt` (JModel state + args), checked by `evaluation/jmaps_eval.py`.
"""
import argparse
import os
import sys

import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("training", "training/grid", "training/data", "training/data/prep", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import jmaps as JM  # noqa: E402
import lane_gt_data as LD  # noqa: E402
import qgraph as QG  # noqa: E402
import trainer as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", default=None, help="the baseline run whose pixel stage is read (default runs/e273_p_s<seed>/model.pt)")
    ap.add_argument("--ctx", default="dil", choices=("none", "dil", "attn"))
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--joint", action="store_true")
    ap.add_argument("--far-w", type=float, default=0.0, help="E304: the heat BCE of the cells beyond 35 m counts 1 + w times")
    ap.add_argument("--raster", default="legacy", choices=("legacy", "exact"), help="E306: the polygon targets (zone, conflict core, stop lines, crossings) rasterised with the exact area rule instead of the legacy half-cell shifted fill")
    ap.add_argument("--sigma", type=float, default=1.5, help="E307: the width (m) of the Gaussian entry / exit heat targets (1.5 = E286)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    T.seed_all(a.seed)
    if a.sigma != 1.5:
        os.environ["BFLC_SIG"] = str(a.sigma)
    if a.raster == "exact":                                                                              # read at the call by `jgt.fill_poly`, inherited by the loader workers
        os.environ["BFLC_RASTER"] = "exact"
    LD.QData.DEFAULT_JUNCTION = True
    model = JM.JModel(width=48, d=a.d, ctx=a.ctx)
    base = a.base or os.path.join(ROOT, "runs", f"e273_p_s{a.seed}", "model.pt")
    n, tot = model.load_base(base)
    ds, dl = LD.loader("train", bs=a.bs, workers=a.workers, shuffle=True, map_mode="train", pixels=a.joint)
    run = T.Run(a.name)
    steps = 20 if a.smoke else a.steps
    ps = list(model.jm.parameters()) + (list(model.pyr.parameters()) if a.joint else [])
    for p in model.parameters():
        p.requires_grad_(False)
    for p in ps:
        p.requires_grad_(True)
    run.log(f"jmaps {a.ctx} joint={a.joint}: {steps} steps, lr {a.lr}, trainable {sum(p.numel() for p in ps) / 1e6:.2f} M, pyramid tensors loaded {n} of {tot} from {base}, train frames {len(ds)} (missing lane GT: {ds.missing})")

    def loss_fn(m, b):
        dev = next(m.parameters()).device
        x = b["x"].to(dev, non_blocking=True).float()
        P, out = m(x, joint=a.joint)
        loss, info = JM.jmaps_loss(out, b, far_w=a.far_w)
        info = {"j_" + k: v for k, v in info.items()}
        if a.joint:
            lp, ip = QG.pixel_loss(P, b)
            loss = loss + lp
            info.update({"px_" + k: v for k, v in ip.items()})
        return loss, info

    T.fit(model, loss_fn, run, steps=steps, loader=dl, lr=a.lr, ckpt_every=500, seed=a.seed, args=vars(a), params=ps, config=dict(design="jmaps", ctx=a.ctx, joint=a.joint, d=a.d, base=base))
    torch.save(dict(state=model.state_dict(), args=vars(a)), run.path("model.pt"))
    run.mark_done()
    run.log("done", a.name)


if __name__ == "__main__":
    main()
