"""Trainer of the lane head (E273): the stages one after the other on the shared harness `training/trainer.py`, each with its own gate; start it through `autoresearch/tools/run.py`.

    wsl bash tools/env/_py14.sh qgraph_train.py --name e273_p_s0 --seed 0 [--phases 2 3 4 5 6 7] [--bs 4] [--smoke]

Phase 2 pixel maps (the 1 m branch starts from the E271 head `runs/e271_a_s<seed>/model.pt`), 3 nodes (pixels frozen), 4 links + Hermite scales (pixels, nodes frozen), 5 sections (the node
head's s output; links frozen), 6 map encoder + assignment (rest frozen), 7 everything jointly at a lower learning rate. Every phase is a run of its own (`runs/<name>_p<k>`, model.pt, .done)
and the next one starts from the previous model; after every phase the gate of the card (kill criterion) is read on 150 VAL frames: a failed gate stops the script with exit code 3 (the run
is a failure, the ledger row says which gate), so that an unattended night does not train on a broken stage. `--smoke`: 20 steps per phase, 8 VAL frames for the gates, gates only reported.
Frames: TRAIN through `lane_gt_data.QData` (the frames whose lane GT is stored); VAL for the gates; TEST never.
"""
import argparse
import json
import os
import sys

import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("training", "training/grid", "training/data", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import lane_gt_data as LD  # noqa: E402
import qgraph as QG  # noqa: E402
import trainer as T  # noqa: E402

STEPS = {2: 3000, 3: 3000, 4: 3000, 5: 1500, 6: 1500, 7: 2000}
LR = {2: 1e-3, 3: 1e-3, 4: 1e-3, 5: 5e-4, 6: 1e-3, 7: 2e-4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phases", type=int, nargs="*", default=[2, 3, 4, 5, 6, 7])
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--head", default=None, help="the E271 head to start from (default runs/e271_a_s<seed>/model.pt)")
    ap.add_argument("--steps", type=int, nargs="*", default=None, help="steps per phase in the order of --phases")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-gates", action="store_true")
    ap.add_argument("--resume-from", default=None, help="a model.pt of an earlier phase to start from (the phases before it are skipped by listing only the later ones)")
    a = ap.parse_args()
    T.seed_all(a.seed)
    model = QG.QGraph(width=a.width)
    hp = a.head or os.path.join(ROOT, "runs", f"e271_a_s{a.seed}", "model.pt")
    if os.path.exists(hp):
        n = model.pyr.load_e271(hp)
    else:
        n = -1
    if a.resume_from:
        model.load_state_dict(torch.load(a.resume_from, map_location="cpu")["state"])
    ds, dl = LD.loader("train", bs=a.bs, workers=a.workers, shuffle=True, map_mode="train")
    steps = dict(zip(a.phases, a.steps)) if a.steps else {k: STEPS[k] for k in a.phases}
    import qgates as QGt
    for k in a.phases:
        sk = 20 if a.smoke else steps[k]
        run = T.Run(f"{a.name}_p{k}")
        run.log(f"phase {k}: {sk} steps, lr {LR[k]}, trainable {sum(p.numel() for p in QG.trainable_params(model, k)) / 1e6:.2f} M, train frames {len(ds)} (missing lane GT: {ds.missing}), "
                f"E271 tensors loaded {n}, seed {a.seed}")
        for p in model.parameters():
            p.requires_grad_(False)
        ps = QG.trainable_params(model, k)
        for p in ps:
            p.requires_grad_(True)
        T.fit(model, lambda m, b, k=k: QG.qgraph_loss(m, b, k), run, steps=sk, loader=dl, lr=LR[k], ckpt_every=500, seed=a.seed, args=vars(a), params=ps, config=dict(design="qgraph", phase=k, width=a.width))
        res = QGt.gate(model, k, run, n=8 if a.smoke else 150)
        run.log("GATE", json.dumps(res, default=float))
        run.status(gate=res)
        if not res.get("pass", True) and not (a.smoke or a.no_gates):
            run.log(f"gate of phase {k} FAILED: {res.get('why')}")
            sys.exit(3)
    run = T.Run(a.name)
    torch.save(dict(state=model.state_dict(), args=vars(a), steps=steps), run.path("model.pt"))
    run.mark_done()
    run.log("all phases done", a.phases)


if __name__ == "__main__":
    main()
