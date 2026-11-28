"""The stage-0 trainer: the lane head B' from random weights, phase by phase, each phase a fit of `trainer.fit` with its own gate (`qgates.py`). A run
starts it through the research worker (`lane_train.py stage0`: the three parts of the champion's chain, one after the other); by hand:

    python -P -m lane_center.training.train.qgraph_train --name <run>/stage0/e292 --seed 0 --phases 3 4 5 6 9 --steps 3000 2500 1000 750 1000 \
        --resume-from runs/e273_p_s0/model.pt --soft-nms-mnms 1.75 0.1 --gate-phases 3 --gate-node-prec 0.75 --gate-node-rec 0.5 [--smoke]

Phases (the parameter groups: `qgraph.PHASE_TRAIN`; the frozen stages run under no_grad: `qgraph.PHASE_FROZEN`; learning rates `LR`, default steps `STEPS`):
1 the E271 grid head (`head.GridHead`, its own network on `grid_data`, batch 8, `losses.grid_loss`; alone: `--phases 1`), 2 pixel maps (the pyramid starts from
the E271 head `--head`, `pyramid.load_e271`), 3 nodes, 4 links + Hermite scales, 5 sections (the node head), 6 map encoder + assignment, 7 everything jointly at
a lower rate, 9 nodes, links, map encoder and assigner jointly on a frozen pyramid (E292). The champion's B' = E271 a (`--phases 1 --steps 4000 --bs 8`) -> E273
p (phases 2 - 7,
`STEPS`, no gate stops) -> E292 a (the command above, from E273's model.pt, with the Matrix-NMS layer; only the gate of phase 3 stops): the blocks of
`configs/base.json` stage0. `--soft-nms-mnms R SIGMA` puts the Matrix-NMS layer (`qmnms.py`) after the node head; it travels with the checkpoint as
args['soft_nms'] = ['mnms', R, sigma] (`qgraph.from_args`). `--resume-from` loads the tensors of an earlier model.pt whose shapes fit.
Folders: <name>/p<k> per phase (the run-directory contract of `trainer.fit`: config.json, log.jsonl, ckpt.pt every 500 steps (phase 1: 400), model.pt, .done);
<name>/model.pt (state + args + steps: the format of runs/e292_a_s<S>, which `lane_net.load_b` reads) and <name>/.done after the last phase.
The step's folder (`step_dir`: the step of the research worker, by hand <name>): its CONTROL file stops the stage (`trainkit.control`: the checkpoint of the
phase is saved, exit 75; the caller clears an old one once per process: `lane_train` before its parts, `train` itself only without a step_dir), and its
log.jsonl gets every row of every phase with `phase` and `part` (the last folder of --name: e271 / e273 / e292) and `step` the step of the phase; the numbers
of a row are keyed <part>.p<phase>.<key> (e292.p3.total, e273.p7.l_link, e292.p3.gate_pass), since steps restart and loss scales differ by phase: a card's
kill rule names such a key, and its after_step counts the steps of that phase.
A restart resumes: a finished phase is loaded from its model.pt (its gate from its status.json; a gate that could not be read is read again), the running one
from its checkpoint, with the batches of its step (`StepBatches`: the frames and the simulated maps (`lane_gt_data.QData(map_seed=)`) are a function of the
seed, the phase and the step; after a roll-back of the loss guard the batches run on from the step reached, so a resume after a roll-back does not see the
frames of the first pass). A folder that holds another recipe is refused, also a finished one.
Gates: after every phase on 150 VAL frames; a failed gate of a phase in `--gate-phases` (default every phase; an empty list: none) ends the stage with exit
code 3 (GATE_EXIT) and no model.pt. `--smoke`: 20 steps per phase, 8 VAL frames, gates only reported. TRAIN frames: `lane_gt_data.QData("train")` (phase 1:
`grid_data.GridData("train")`), VAL for the gates, TEST never. Test: net/tests/test_qgraph_train.py"""
import argparse
import json
import os

import torch

from trainkit import control
from lane_center.net.loss import loss_qgraph as LQG
from lane_center.net.loss import losses as GL
from lane_center.net.model import head as GH
from lane_center.net.model import lane_net as LN
from lane_center.net.model import qgraph as QG
from lane_center.net.prep import grid_data as GD
from lane_center.net.prep import lane_gt_data as LD
from lane_center.training import bflc_paths as BP
from lane_center.training import trainer as T
from lane_center.training.train import qgates as QGt

STEPS = {1: 4000, 2: 3000, 3: 6000, 4: 5000, 5: 2000, 6: 1500, 7: 2000, 9: 1000}
LR = {1: 1e-3, 2: 1e-3, 3: 1e-3, 4: 1e-3, 5: 5e-4, 6: 1e-3, 7: 2e-4, 9: 2e-4}
CKPT_EVERY = {1: 400}                                                       # the other phases: 500
GATE_EXIT = 3                                                               # the exit code of a failed gate
# what makes the weights (the gates may change on a resume, as E273 did)
RECIPE = ("seed", "phases", "steps", "bs", "width", "head", "resume_from", "soft_nms", "smoke")


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phases", type=int, nargs="*", default=[2, 3, 4, 5, 6, 7])
    ap.add_argument("--steps", type=int, nargs="*", default=None, help="steps per phase in the order of --phases (default STEPS)")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--head", default=None, help="the E271 grid head (model.pt) the pyramid starts from and the gates of phases 2 and 3 compare with (default runs/e271_a_s<seed>/model.pt)")
    ap.add_argument("--resume-from", default=None, dest="resume_from", help="a model.pt of an earlier part to start from (its tensors whose shapes fit)")
    ap.add_argument("--soft-nms-mnms", type=float, nargs=2, default=None, dest="soft_nms_mnms", metavar=("RADIUS", "SIGMA"),
                    help="the Matrix-NMS layer after the node head (E292)")
    ap.add_argument("--gate-phases", type=int, nargs="*", default=None, dest="gate_phases", help="only a failed gate of these phases stops the stage (default: every phase; an empty list: none)")
    ap.add_argument("--gate-node-prec", type=float, default=None, dest="gate_node_prec", help="the node gate also needs this node precision within 2 m")
    ap.add_argument("--gate-node-rec", type=float, default=None, dest="gate_node_rec", help="the node gate also needs this node recall within 2 m")
    ap.add_argument("--smoke", action="store_true", help="20 steps per phase, 8 VAL frames, gates only reported")
    a = ap.parse_args(argv)
    a.steps = [20] * len(a.phases) if a.smoke else (a.steps or [STEPS[k] for k in a.phases])
    assert len(a.steps) == len(a.phases) and len(set(a.phases)) == len(a.phases), f"--steps {a.steps}: one value per phase of --phases {a.phases}, each phase once"
    assert set(a.phases) <= set(STEPS) and (a.phases == [1] or 1 not in a.phases), f"--phases {a.phases}: of {sorted(STEPS)}; phase 1 (the E271 grid head, another network) alone"
    a.head = a.head or f"runs/e271_a_s{a.seed}/model.pt"
    a.soft_nms = ["mnms"] + [float(v) for v in a.soft_nms_mnms] if a.soft_nms_mnms else None
    return a


class StepBatches(torch.utils.data.Sampler):
    """the batches of a phase by step, from step `start` on, without end: epoch e = the frames in the order of a permutation drawn from (seed, e), `bs` of them
    per step, the incomplete last batch of an epoch dropped; a resumed phase continues with the batch of its step (after a roll-back of the loss guard the
    batches run on: the step of the batches is then ahead of the trainer's). keyed: an item is (step, frame), the key of its simulated map
    (`lane_gt_data.QData(map_seed=)`)"""

    def __init__(self, n, bs, seed, start=0, keyed=False):
        self.n, self.bs, self.seed, self.start, self.keyed = n, bs, seed, start, keyed

    def epoch(self, e):
        return torch.randperm(self.n, generator=torch.Generator().manual_seed(self.seed * 100003 + e)).tolist()

    def __iter__(self):
        per, step, cur, perm = self.n // self.bs, self.start, None, None
        while True:
            e, i = divmod(step, per)
            if e != cur:
                cur, perm = e, self.epoch(e)
            frames = perm[i * self.bs:(i + 1) * self.bs]
            yield [(step, f) for f in frames] if self.keyed else frames
            step += 1


class PhaseRun(T.Run):
    """the folder of one phase (<name>/p<k>); its log.jsonl rows go also to the log.jsonl of the step (the kill rules of the research worker read that one),
    their numbers keyed <part>.p<phase>.<key> (module docstring)"""

    def __init__(self, name, root, step, part, phase):
        super().__init__(name, root)
        self.step, self.part, self.phase = step, part, phase

    def jsonl(self, row):
        super().jsonl(row)
        step_row(self.step, self.part, self.phase, row)


def step_row(step, part, phase, row):
    """a row of the step's log.jsonl: step (of the phase), part, phase and the numbers of `row` as <part>.p<phase>.<key>"""
    step.jsonl(dict({f"{part}.p{phase}.{k}": v for k, v in row.items() if k != "step"}, step=row["step"], part=part, phase=phase))


def check_start(args, folder):
    """before anything is written: a folder that already holds a part (also a finished one) is only RESUMED (the same recipe), never another one's"""
    p = os.path.join(folder, "recipe.json")
    if os.path.exists(p):
        old = json.load(open(p, encoding="utf-8"))
        assert old == {k: args[k] for k in RECIPE}, f"{folder} holds another recipe ({old}): use another --name"


def build(a):
    """the network of the part: the E271 grid head (phase 1) or B' (its pyramid from the E271 head, then the tensors of --resume-from whose shapes fit)"""
    T.seed_all(a.seed)
    if a.phases == [1]:
        return GH.GridHead(width=a.width), "E271 grid head"
    model = LN.build_b(dict(width=a.width, soft_nms=a.soft_nms))
    n = model.pyr.load_e271(BP.resolve(a.head))
    what = f"B' (width {a.width}, Matrix-NMS {a.soft_nms}), pyramid from the E271 head {a.head} ({n} tensors)"
    if a.resume_from:
        sd, own = LN.load_checkpoint(BP.resolve(a.resume_from))["state"], model.state_dict()
        fit = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(fit, strict=False)
        what += f", {len(fit)} of {len(own)} tensors from {a.resume_from} (not loaded: {sorted(set(own) - set(fit))[:6]})"
    return model, what


def phase_loss(k, dev):
    if k == 1:
        return lambda m, b: GL.grid_loss(m(b["x"].to(dev, non_blocking=True).float()), {n: b[n].to(dev) for n in ("g05", "t05", "m05")}, per100=GL.PER100)
    return lambda m, b: LQG.qgraph_loss(m, b, k)


def phase_params(model, k):
    """the tensors phase k trains (every other one frozen: requires_grad off)"""
    if k == 1:
        return list(model.parameters())
    for p in model.parameters():
        p.requires_grad_(False)
    ps = QG.trainable_params(model, k)
    for p in ps:
        p.requires_grad_(True)
    return ps


def stopped(step, where):
    """a stop request of the step's folder between two phases: exit 75 (the phases before are finished and saved)"""
    reason = control.stop_requested(step.dir)
    if reason is None:
        return
    control.mark_stopped(step.dir, 0, reason)
    step.log(f"stop requested ({reason}) before phase {where}: exit {control.EXIT_STOPPED}")
    raise SystemExit(control.EXIT_STOPPED)


def train(a, runs=T.RUNS, step_dir=None, gate_n=None, fit_n=None, workers=None):
    """the phases of the parsed arguments `a` (no GPU check: `main` has it) -> 0, or SystemExit(GATE_EXIT) / SystemExit(75). step_dir = the folder whose CONTROL
    stops the stage and whose log.jsonl gets every row (default the part's own folder); gate_n / fit_n / workers: the VAL frames of a gate, the TRAIN frames of
    the E271 threshold, the loader workers (default 150 (smoke 8) / 300 (smoke 16) / --workers; a test sets them)"""
    args = dict(vars(a))
    folder = os.path.join(runs, a.name)
    check_start(args, folder)
    if os.path.exists(os.path.join(folder, ".done")):
        print(f"{folder} is complete")
        return 0
    part, step = T.Run(a.name, root=runs), T.Run(step_dir or folder)
    with open(part.path("recipe.json"), "w", encoding="utf-8") as fh:
        json.dump({k: args[k] for k in RECIPE}, fh)
    if step_dir is None:
        # by hand: once per process (a caller with a step folder clears it once itself)
        control.clear_stale(step.dir)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gate_n, fit_n = gate_n or (8 if a.smoke else 150), fit_n or (16 if a.smoke else QGt.FIT_FRAMES)
    workers = a.workers if workers is None else workers
    model, what = build(a)
    ds = GD.GridData("train") if a.phases == [1] else LD.QData("train", map_mode="train", map_seed=0)
    part.log(f"stage 0 {a.name}: {what}; train frames {len(ds)}" + (f" (missing lane GT: {ds.missing})" if hasattr(ds, "missing") else "") + f", seed {a.seed}")
    name = os.path.basename(os.path.normpath(a.name))
    gate = lambda k, log: QGt.gate(model.to(dev), k, BP.resolve(a.head), gate_n, a.gate_node_prec, a.gate_node_rec, log, workers, fit_n)  # noqa: E731
    for k, sk in zip(a.phases, a.steps):
        stopped(step, k)
        run = PhaseRun(f"{a.name}/p{k}", runs, step, name, k)
        if os.path.exists(run.path(".done")):
            model.load_state_dict(LN.load_checkpoint(run.path("model.pt"))["state"])
            res = run._status.get("gate")
            # never read, or not readable then (an OOM in the gate): read again
            if res is None or str(res.get("why", "")).startswith(QGt.UNREAD):
                res = gate(k, run.log)
                run.status(gate=res)
            run.log(f"phase {k} is complete: loaded")
        else:
            ps = phase_params(model, k)
            ck = run.path("ckpt.pt")
            start = LN.load_checkpoint(ck)["step"] if os.path.exists(ck) else 0                  # a resumed phase: the batches from its step on
            order = StepBatches(len(ds), a.bs, a.seed * 10 + k, start, keyed=k != 1)
            if k != 1:
                # the simulated maps of the phase: a function of (seed, phase, step, frame)
                ds.map_seed = a.seed * 10 + k
            dl = (GD.loader(ds, workers=workers, batches=order) if k == 1 else
                  torch.utils.data.DataLoader(ds, batch_sampler=order, num_workers=workers, collate_fn=LD.collate, persistent_workers=workers > 0, prefetch_factor=2 if workers > 0 else None))
            run.log(f"phase {k}: {sk} steps from step {start}, lr {LR[k]}, batch {a.bs}, trainable {sum(p.numel() for p in ps) / 1e6:.3f} M, seed {a.seed}")
            T.fit(model, phase_loss(k, dev), run, steps=sk, loader=dl, lr=LR[k], ckpt_every=CKPT_EVERY.get(k, 500), seed=a.seed, args=args, params=ps, device=dev,
                  config=dict(design="qgraph", phase=k, width=a.width), control_dir=step.dir)
            res = gate(k, run.log)
            run.status(gate=res)
        run.log("GATE", json.dumps(res, default=float))
        step_row(step, name, k, dict(step=sk, gate_pass=float(bool(res.get("pass", True)))))
        if not res.get("pass", True) and not a.smoke and (a.gate_phases is None or k in a.gate_phases):
            run.log(f"gate of phase {k} FAILED: {res.get('why')}; exit {GATE_EXIT}")
            raise SystemExit(GATE_EXIT)
    torch.save(dict(state=model.state_dict(), args=args, steps=dict(zip(a.phases, a.steps))), part.path("model.pt"))
    part.mark_done()
    part.log("all phases done", a.phases)
    return 0


def main(argv=None, runs=T.RUNS, step_dir=None):
    a = parse_args(argv)
    assert a.smoke or torch.cuda.is_available(), "no GPU visible: stage 0 trains on the GPU (bf16); only --smoke may run on the CPU"
    return train(a, runs, step_dir)


if __name__ == "__main__":
    raise SystemExit(main())
