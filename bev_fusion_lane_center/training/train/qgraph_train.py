"""Trainer of the lane head (E273): the stages one after the other on the shared harness `training/trainer.py`, each with its own gate; start it through `autoresearch/tools/run.py`.

    wsl bash tools/env/_py14.sh qgraph_train.py --name e273_p_s0 --seed 0 [--phases 2 3 4 5 6 7] [--bs 4] [--smoke]

E274 switches (`--ev-w`, `--node-w-evt`, `--node-w-inter`, `--long-match`, see `qgraph.CFG`) default to the E273 loss; `--gate-phases` lets only the named phases stop the script.
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

STEPS = {2: 3000, 3: 6000, 4: 5000, 5: 2000, 6: 1500, 7: 2000}
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
    ap.add_argument("--gate-phases", type=int, nargs="*", default=None, help="only a failed gate of these phases stops the script (default: every phase; an empty list = none, like --no-gates)")
    ap.add_argument("--ev-w", type=float, default=1.0, help="E274: weight of the 1 m event heat-map loss")
    ap.add_argument("--gate-node-prec", type=float, default=None, help="E274: the node gate also needs this node precision within 2 m (scorer definition)")
    ap.add_argument("--gate-node-rec", type=float, default=None, help="E274: the node gate also needs this node recall within 2 m")
    ap.add_argument("--node-w-evt", type=float, default=0.0, help="E274: extra weight of the existence positives at event cells")
    ap.add_argument("--node-w-inter", type=float, default=0.0, help="E274: extra weight of the existence positives inside the grown intersection mask")
    ap.add_argument("--long-match", type=float, default=0.0, help="E274: weight of the dual-softmax loss over the long-link candidates")
    ap.add_argument("--tier-w", type=float, nargs=3, default=None, help="E276: factor of the positive node terms at a GT node of tier 1 / 2 / 3 (needs the l4 lane GT)")
    ap.add_argument("--vis-w", type=float, default=1.0, help="E276: extra factor for GT nodes that no camera sees (0 = not demanded)")
    ap.add_argument("--neg-w", type=float, nargs=4, default=None, help="E276: factor of the negatives of the existence loss by zone: tier 1, tier 2, tier 3, no lane")
    ap.add_argument("--link-tier-w", type=float, nargs=3, default=None, help="E276: factor of the link losses by the tier of the source node")
    ap.add_argument("--sparse", type=float, nargs=2, default=None, metavar=("EPS", "CHORD"), help="E276: the sparse node target: deviation (m) and longest chord (m)")
    ap.add_argument("--resume-from", default=None, help="a model.pt of an earlier phase to start from (the phases before it are skipped by listing only the later ones)")
    a = ap.parse_args()
    T.seed_all(a.seed)
    QG.CFG.update(ev_w=a.ev_w, node_w_evt=a.node_w_evt, node_w_inter=a.node_w_inter, long_match=a.long_match)               # E274 switches (defaults = E273)
    QG.CFG.update(tier_w=a.tier_w, vis_w=a.vis_w, neg_w=a.neg_w, link_tier_w=a.link_tier_w)                                 # E276 switches (defaults = E273)
    if a.sparse:
        LD.QData.DEFAULT_SPARSE = tuple(a.sparse)
    if (a.tier_w or a.neg_w or a.link_tier_w or a.vis_w != 1.0):
        from autogt import lane_gt                                                                                          # the tier switches need the l4 labels: refuse an l3 store
        assert lane_gt.version_of(lane_gt.signature()) not in (None, "l1", "l2", "l3"), "the tier switches need the lane GT l4 (this source is " + str(lane_gt.version_of(lane_gt.signature())) + ")"
    model = QG.QGraph(width=a.width)
    hp = a.head or os.path.join(ROOT, "runs", f"e271_a_s{a.seed}", "model.pt")
    if os.path.exists(hp):
        n = model.pyr.load_e271(hp)
    else:
        n = -1
    if a.resume_from:                                                                       # the tensors of an earlier phase model whose shapes still fit (a changed stage keeps its fresh start)
        sd, own = torch.load(a.resume_from, map_location="cpu")["state"], model.state_dict()
        fit = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(fit, strict=False)
        print(f"resume-from {a.resume_from}: {len(fit)} of {len(own)} tensors loaded; not loaded: {sorted(set(own) - set(fit))[:6]}...", flush=True)
    ds, dl = LD.loader("train", bs=a.bs, workers=a.workers, shuffle=True, map_mode="train")
    steps = dict(zip(a.phases, a.steps)) if a.steps else {k: STEPS[k] for k in a.phases}
    import qgates as QGt
    tier_arm = bool(a.tier_w or a.neg_w or a.link_tier_w or a.vis_w != 1.0 or a.sparse)
    QGt.GATE_CFG.update(node_prec_min=a.gate_node_prec, node_rec_min=a.gate_node_rec, tier_mode=tier_arm)
    assert not tier_arm or ds.missing == 0, f"an E276 arm needs the complete lane GT store: {ds.missing} training frames have no stored graph (the loader would silently drop them)"
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
        T.fit(model, lambda m, b, k=k: QG.qgraph_loss(m, b, k), run, steps=sk, loader=dl, lr=LR[k], ckpt_every=500, seed=a.seed, args=vars(a), params=ps, config=dict(design="qgraph", phase=k, width=a.width, cfg=dict(QG.CFG)))
        res = QGt.gate(model, k, run, n=8 if a.smoke else 150)
        run.log("GATE", json.dumps(res, default=float))
        run.status(gate=res)
        if not res.get("pass", True) and not (a.smoke or a.no_gates) and (a.gate_phases is None or k in a.gate_phases):
            run.log(f"gate of phase {k} FAILED: {res.get('why')}")
            sys.exit(3)
    run = T.Run(a.name)
    torch.save(dict(state=model.state_dict(), args=vars(a), steps=steps), run.path("model.pt"))
    run.mark_done()
    run.log("all phases done", a.phases)


if __name__ == "__main__":
    main()
