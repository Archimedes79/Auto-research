"""Trainer of E372 (stage S4 of the token champion, D113): the JOINT fine-tune of the champion token head with B' in the loop (`data/lane_live_tok.LiveSource`: B''s tokens and the labels made live in every step). Start it through `autoresearch/tools/run.py`.

    wsl bash tools/env/_pyr.sh jlive_train.py --name e372_s4_s0 --seed 0 --head-run e373_cb_s0 --train-nodes 1 --steps 3000 --bs 8 --lr 1e-4 --workers 4 [--live-inputs 1] [--b-run runs/e292_a_s0] [--b-weight 1.0] [--none-weight W] [--pos-loss sum|max] [--head-seed N] [--b-state-from R] [--train-pyr 1 --pyr-lr-mult 0.1]

Arms (card section 0): z = `--train-nodes 0` (B' frozen, the head's loss only), s4 = `--train-nodes 1` (B''s node head trains on the head's loss + b_weight x its own node loss `lane_live_tok.b_loss`), b = `--train-nodes 1 --live-inputs 0` (the same trainer, the head cuts its inputs:
B''s node head trains on its own loss only). `--seed` = B''s seed (the default `--b-run` runs/e292_a_s<S>, the frame order of the E365 cache; the head run must be of the same seed), `--head-seed` = the data order (default: the head run's).
One step: `no, tk = src.tokens(x)`; the head on `head_inputs(tk)` (queued before the labels: the overlap of lane_live_tok's PLAN); `src.labelled(tk, ..., token_drop=0, gate=False)`; loss = `jtoken.jtoken_loss` with the head run's options (none weight, pos loss, none weight near;
`--none-weight` / `--pos-loss` override; the E392 options rounds / msg / wire weight are the head run's, no flag: the head is built with them, in the recipe and in model.pt's args) + b_weight x B''s node loss (s4, b; computed in float32 outside the autocast, logged in every arm, under no_grad in z). The head runs in fit's bf16 autocast as in S1, B' in float32 (LiveSource); no cudnn / TF32 flag is set.
`trainer.fit` trains `Joint` (the head + the LiveSource): its optimiser gets the head and B''s node head (the frozen tensors are left out), one cosine schedule (5 % warm-up), the same lr for both; its checkpoint holds both (resume, roll-back) and the record of the kill rules (extra state).
fit's joint clip is OFF (clip=0: it only measures, `grad_norm` of the log); the CLIP IS PER GROUP: the head's gradients at 3.0 on their own, the node head's at 3.0 on their own (B''s node gradient is 8.7 - 17.6 against the head's ~0.5: one clip would scale the head's step
by ~0.17 in s4 only), done in `StepHook`, torch's global optimiser pre-step hook (fit keeps its optimiser to itself: the hook runs after fit's backward, right before AdamW's step). The kill rules (card), each stops the run with `Kill` (non-zero exit):
  live        the head's `live_inputs` is the arm's in every step (s4: a silent False would train B' on its own loss only; b: a silent True would let the head's loss reach B); at the first step of a process the head's loss gives the node head a gradient of norm > 0 (s4)
              or exactly 0 (b); at the first update of a process no tensor of B' outside the trained node head (z: none of B') holds a gradient
  finite      the head's loss and B''s node loss are finite in every step (fit alone would skip such a step)
  overflow    a frame over `lane_tokcache.PMAX` target pairs (`LabelOverflow`): the step is skipped (a NaN loss: fit's skip, no optimiser step) and counted; kill when the count exceeds 1 % of the steps done, from step 100 on
  B rise      (s4, b) the mean of B''s node loss over the most recent 200 steps above 1.2 x its mean over steps 1 - 200, checked at every step after step 200
  learning    the head's update norm (AdamW's step from its state, the decoupled weight decay left out) finite and > 0 on every logged step; the mean of the HEAD's loss over steps 1500 - 2000 at most 1.10 x its mean over steps 1 - 200 (every arm)
A kill writes its reason into the file KILLED and status.json (`killed`); such a folder is never resumed.
Logs (every logged step, fit's row of log.jsonl): the head's loss parts, `head` (the head's loss), `b_loss` and its parts b_*, b_start / b_mean (the start mean and the running mean of B''s loss), h_start, overflow (the count), clip_head / clip_nodes (the two clip factors), upd_head (the
head's update norm); over the last 100 steps of the process: label_s (seconds of the label loop per step) and the moving labels pair_cov (in-gate GT pairs / GT links), n_valid (valid tokens per frame), gt_cov (GT nodes with a token / GT nodes); of the step: nn_min (the smallest
distance in m between two valid live tokens of a frame) and pos_grad_max (the largest |gradient| of the head's loss on the live token positions; s4 only, NaN in z and b); s4 every 200 steps and at the last step: ushare (the one-step
preconditioned share |g_head / (sqrt(v_hat) + eps)| / (that + |g_b / (sqrt(v_hat) + eps)|) on the node head, g_b = the gradient of b_weight x B''s loss, g_head = the full gradient - g_b, v_hat = AdamW's second moment of the state) and gshare (the same with the raw gradient norms).
Pairing: the frame tokens of the first 5 batches are logged; a resume and a roll-back of fit's guard (lr x 0.7) are logged and listed in status.json (`resumed_at`, `rollbacks_at`): the pair with the twin arm is broken from there.
model.pt: the head in the format of `jtoken_train` (state, args with live_inputs: `jtoken_train.load_model` rebuilds it) + b_state (the complete `QGraph` state of B' after training) + b_args (the B' run's args) + rec; `load_b` reads B'. It is written, read back and checked
(the head and B' as trained, bit for bit; B' outside the node head, and in z all of B', as at the start); a failure renames fit's .done to .done.FALSE and sets status.json `state` failed. A folder that holds another recipe is refused; the same recipe is resumed. `main` needs a GPU (no smoke run: the
50-step gate runs the real data); `train(a, runs, ckpt_every)` is the loop without that check (`ckpt_every` TEST-ONLY). Tests: grid/test_jlive_train.py

E393 (the pyramid trained by the token loss): `--b-state-from R` starts B' from R's b_state (R made on the same base B' file `--b-run`, whose sha256 still checks the cache; R's file sha256 in the recipe); `--train-pyr 1` trains B''s pyramid too
(`LiveSource(train_pyr=)`: out of no_grad, no pixel loss) in its own AdamW group at `--pyr-lr-mult` x the schedule's lr (`trainer.fit`'s per-group lr_mult) and its own clip group (CLIP); at the first step its gradient from the head's loss must be
finite and > 0 (Kill), the tensors it does not reach are logged; logged steps add gn_head / gn_nodes / gn_pyr (before the clip), pshare (the pyramid's share of the three), clip_pyr, upd_nodes / upd_pyr. Two kills of every arm on the per-step
record cov / nact (GT-node coverage; ACTIVE cells per frame: existence above P_ACTIVE before the top-K cut, uncapped, since the valid tokens saturate at K): the mean of the last 200 steps below 0.95 x / above 1.25 x the mean of steps 1 - 200 (cov_start / cov_mean / nact_start / nact_mean in the log). The B-rise kill acts only where B''s node
loss is in the loss (train_nodes and b_weight > 0). The export lets `pyr.*` move with train_pyr; the base of every check is the start state (after --b-state-from).
"""
import argparse
import collections
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.optim.optimizer import register_optimizer_step_pre_hook

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub_ in ("training", "training/grid", "training/data", "training/data/prep", "training/train", "evaluation"):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import jtoken_train as JKT  # noqa: E402
import lane_live_tok as LLT  # noqa: E402
import qgraph as QG  # noqa: E402
import trainer as T  # noqa: E402

RECIPE = ("seed", "head_run", "head_sha256", "b_run", "b_sha256", "train_nodes", "live_inputs", "b_weight", "none_weight", "pos_loss", "head_seed", "bs", "steps", "lr", "rounds", "wire_weight", "msg",
          "b_state_from", "b_state_sha256", "train_pyr", "pyr_lr_mult")
RECIPE_DEFAULTS = dict(b_state_from=None, b_state_sha256=None, train_pyr=0, pyr_lr_mult=1.0)        # E393's keys as a config of before them reads (the other keys: `jtoken_train.ARCH_DEFAULTS`)
K = 512                     # tokens per frame, as the cache
CLIP = 3.0                  # the clip of each group (the head, the node head)
LOG_EVERY = 100             # fit's logged steps
WIN = 200                   # the start window (steps 1 - 200) of B''s and the head's loss, and the running window of B''s loss
B_RISE = 1.2                # kill: B''s node loss of the most recent WIN steps above B_RISE x its start mean (where it is in the loss: train_nodes and b_weight > 0)
COV_DROP = 0.95             # kill (E393, every arm): the GT-node coverage of the most recent WIN steps below COV_DROP x its mean over steps 1 - WIN
NACT_RISE = 1.25            # kill (E393, every arm): the ACTIVE cells per frame (existence above lane_tokcache.P_ACTIVE on the whole 2 m map, before the top-K cut: uncapped; the valid tokens are this count capped at K, so they cannot show a blow-up past the budget) of the most recent WIN steps above NACT_RISE x its mean over steps 1 - WIN
H_LATE, H_RISE = (1500, 2000), 1.10                 # kill: the head's loss over steps 1500 - 2000 above H_RISE x its start mean (every arm)
OVERFLOW_MAX, OVERFLOW_FROM = 0.01, 100             # kill: more LabelOverflow steps than 1 % of the steps done, from step 100 on
SHARE_EVERY = 200           # s4: the update share every 200 steps (and at the last step)
FRAMES_LOGGED = 5           # the frame tokens of the first 5 batches are logged (the pairing of the arms)


class Kill(RuntimeError):
    """a kill rule of the card fired (module docstring): the run stops with a non-zero exit"""


def arm(train_nodes, live_inputs):
    return "z" if not train_nodes else ("s4" if live_inputs else "b")


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0, help="B''s seed: the default --b-run, the frame order of its E365 cache; the head run must be of this seed")
    ap.add_argument("--head-run", required=True, dest="head_run", help="the champion head under runs/ (e.g. e373_cb_s0): the start weights and the loss options")
    ap.add_argument("--b-run", default=None, dest="b_run", help="B''s run folder relative to the problem folder (default runs/<the head run's ident>_s<seed>)")
    ap.add_argument("--train-nodes", type=int, choices=(0, 1), required=True, dest="train_nodes", help="1: B''s node head trains (arms s4, b); 0: B' frozen (arm z)")
    ap.add_argument("--live-inputs", type=int, choices=(0, 1), default=1, dest="live_inputs", help="1: the head keeps its inputs' graph (s4); 0: it cuts them (arm b with --train-nodes 1)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--b-weight", type=float, default=1.0, dest="b_weight", help="the weight of B''s node loss in the loss (s4, b)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--none-weight", type=float, default=None, dest="none_weight", help="default: the head run's")
    ap.add_argument("--pos-loss", choices=("sum", "max"), default=None, dest="pos_loss", help="default: the head run's")
    ap.add_argument("--head-seed", type=int, default=None, dest="head_seed", help="the data order (default: the head run's head seed)")
    ap.add_argument("--b-state-from", default=None, dest="b_state_from", help="E393: a jlive run under runs/ whose b_state is the start B' (its base B' file must be --b-run's); default: --b-run's own state")
    ap.add_argument("--train-pyr", type=int, choices=(0, 1), default=0, dest="train_pyr", help="E393: 1 = B''s pyramid trains too (on the loss of the step; it has no pixel loss here); needs --train-nodes 1 --live-inputs 1")
    ap.add_argument("--pyr-lr-mult", type=float, default=1.0, dest="pyr_lr_mult", help="E393: the pyramid's lr = this x the schedule's (its own AdamW group); only with --train-pyr 1")
    a = ap.parse_args(argv)
    assert a.steps > 0 and a.bs > 0 and a.lr > 0 and a.b_weight >= 0, f"--steps {a.steps} --bs {a.bs} --lr {a.lr} --b-weight {a.b_weight}"
    assert not a.train_pyr or (a.train_nodes and a.live_inputs), "--train-pyr 1 needs --train-nodes 1 --live-inputs 1 (the pyramid learns from the head's loss through the node head)"
    assert a.pyr_lr_mult > 0 and (a.train_pyr or a.pyr_lr_mult == 1.0), f"--pyr-lr-mult {a.pyr_lr_mult}: > 0, and only with --train-pyr 1"
    return a


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def resolve(a, runs):
    """the run's args: the CLI + the head run's loss options and its E392 head options (rounds, wire_weight, msg: the head of the run is built and its early-read loss is added as the head run's; its removed options checked: `jtoken_train.check_removed`;
    `build_head` builds the champion architecture) + the paths and sha256 of both start checkpoints"""
    h = torch.load(os.path.join(runs, a.head_run, "model.pt"), map_location="cpu", weights_only=False)["args"]
    g = lambda k: h[k] if h.get(k) is not None else JKT.ARCH_DEFAULTS.get(k)  # noqa: E731   (a key stored as None means the default, as `jtoken_train.arch` reads it)
    assert h.get("seed") == a.seed, f"--seed {a.seed}: the head run {a.head_run} learned on the tokens of B' seed {h.get('seed')}"
    JKT.check_removed(h, f"the head run {a.head_run}")
    assert not h.get("pos_head"), f"the head run {a.head_run} has a position head (E395): the stage-2 trainer neither builds it nor trains its loss (no GT geometry in the live batch)"
    assert h.get("K") in (None, K) and not h.get("cache"), f"the head run {a.head_run} was trained on K {h.get('K')} / cache root {h.get('cache')}: this trainer builds the tokens of the default cache at K {K}"
    b_run = a.b_run or f"runs/{h.get('ident') or 'e292_a'}_s{a.seed}"
    b_sha = JKT.file_sha256(os.path.join(_abs(b_run), "model.pt"))
    s_sha = None
    if a.b_state_from:                                                                          # E393: the start B' = the b_state of a jlive run made on the same base B' file (the architecture, the cache check)
        p = os.path.join(runs, a.b_state_from, "model.pt")
        s = torch.load(p, map_location="cpu", weights_only=False)
        assert "b_state" in s, f"--b-state-from {a.b_state_from}: its model.pt carries no b_state"
        assert s["args"].get("b_sha256") == b_sha, f"--b-state-from {a.b_state_from}: its base B' file (b_sha256 {str(s['args'].get('b_sha256'))[:12]}) is not --b-run {b_run} ({b_sha[:12]})"
        s_sha = JKT.file_sha256(p)
    return dict(vars(a), ident=h.get("ident") or "e292_a", cache=h.get("cache"), K=K,
                none_weight=g("none_weight") if a.none_weight is None else a.none_weight, pos_loss=g("pos_loss") if a.pos_loss is None else a.pos_loss,
                head_seed=a.head_seed if a.head_seed is not None else (h["seed"] if h.get("head_seed") is None else h["head_seed"]),
                rounds=g("rounds"), wire_weight=g("wire_weight"), msg=g("msg"),
                b_run=b_run, head_sha256=JKT.file_sha256(os.path.join(runs, a.head_run, "model.pt")), b_sha256=b_sha, b_state_sha256=s_sha)


def check_start(args, folder):
    """before anything is written: a folder that a kill rule stopped (its KILLED file) is never resumed; a folder that already holds a run is only RESUMED (the same recipe), never another run's"""
    killed = os.path.join(folder, "KILLED")
    assert not os.path.exists(killed), f"{folder} was stopped by a kill rule ({open(killed, encoding='utf-8').read().strip()}): it is not resumed, use another --name"
    held = [f for f in ("config.json", "ckpt.pt", "model.pt") if os.path.exists(os.path.join(folder, f))]
    if held:
        c = json.load(open(os.path.join(folder, "config.json"))) if "config.json" in held else {}
        old = {k: (c.get("args") or {}).get(k, RECIPE_DEFAULTS.get(k, JKT.ARCH_DEFAULTS.get(k))) for k in RECIPE}                    # a run of before an option has its default
        assert c.get("design") == "jlive" and old == {k: args.get(k) for k in RECIPE}, f"{folder} holds {held} of another run ({c.get('design')}, {old}): use another --name"


class Joint(nn.Module):
    """what `trainer.fit` trains: the token head + the live source (B'). `rec` = the record of the kill rules (b / h: B''s node loss and the head's loss of every step done, NaN on a skipped one; overflow: the steps skipped by LabelOverflow;
    E393: cov / nact: the GT-node coverage (GT nodes with a token / GT nodes of the batch, NaN without GT) and the active cells per frame (existence above P_ACTIVE, before the top-K cut) of every step done, NaN on a skipped one) rides in the
    checkpoint as extra state, so that a resume and a roll-back keep the rules right (a checkpoint of before cov / nact: empty lists, the start window of the two kills is then the 200 steps after the resume)"""

    def __init__(self, head, src):
        super().__init__()
        self.head, self.src = head, src
        self.rec = dict(b=[], h=[], overflow=[], cov=[], nact=[])

    def get_extra_state(self):
        return {k: list(v) for k, v in self.rec.items()}

    def set_extra_state(self, s):
        self.rec = dict(dict(cov=[], nact=[]), **{k: list(v) for k, v in s.items()})


def build(args, runs):
    """-> (Joint: the head of the head run with the arm's live_inputs and its weights, B' of the B' run with the node head trainable in s4 / b (and its pyramid with train_pyr), its state the b_state of `b_state_from` when given (strict); B''s run args)"""
    head = JKT.build_head(args)
    head.load_state_dict(torch.load(os.path.join(runs, args["head_run"], "model.pt"), map_location="cpu", weights_only=False)["state"])
    b_args = torch.load(os.path.join(_abs(args["b_run"]), "model.pt"), map_location="cpu", weights_only=False).get("args") or {}
    src = LLT.LiveSource(args["b_run"], K, train_nodes=bool(args["train_nodes"]), train_pyr=bool(args.get("train_pyr")))
    if args.get("b_state_from"):
        p = os.path.join(runs, args["b_state_from"], "model.pt")
        assert JKT.file_sha256(p) == args["b_state_sha256"], f"{p} changed since the run was resolved"
        src.model.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)["b_state"])
    return Joint(head, src), b_args


def batches(seed, bs, workers, b_run, ident):
    """the TRAIN batches of the live source (`lane_live_tok.loader`: the frames of the E365 cache of B' seed `seed`, in its order) after the check that `b_run`/model.pt is the checkpoint that cache was made from (the head learned on its tokens)"""
    LLT.check_run_sha(LLT.LP.cache_dir("train", seed, ident), b_run)
    return LLT.loader("train", seed, bs, workers, ident=ident)[1]


# ------------------------------------------------------------------------------------------------------------------------------ the rules
def _mean(xs):
    xs = np.asarray(xs, float)
    return float(np.nanmean(xs)) if xs.size and not np.isnan(xs).all() else float("nan")


def _norm(ts):
    return math.sqrt(sum(float(t.double().square().sum()) for t in ts if t is not None))


def verdict(rec, b_rise):
    """the card's kill rules on the record of the steps done (`Joint.rec`) -> None or the reason (module docstring); `b_rise`: B''s node loss is in the loss (train_nodes and b_weight > 0), so its rise is a kill (otherwise only logged)"""
    b, h, n = rec["b"], rec["h"], len(rec["overflow"])
    cov, na = rec.get("cov", []), rec.get("nact", [])
    done = len(b)
    if done >= OVERFLOW_FROM and n > OVERFLOW_MAX * done:
        return f"{n} of the {done} steps hit LabelOverflow (more than {OVERFLOW_MAX:.0%} of the steps)"
    if b_rise and done > WIN and _mean(b[-WIN:]) > B_RISE * _mean(b[:WIN]):
        return f"B''s node loss rose: its mean over the last {WIN} steps {_mean(b[-WIN:]):.4f} > {B_RISE} x its mean over steps 1 - {WIN} {_mean(b[:WIN]):.4f} (B is being destroyed)"
    if len(cov) > WIN and _mean(cov[-WIN:]) < COV_DROP * _mean(cov[:WIN]):
        return f"the GT-node coverage fell: its mean over the last {WIN} steps {_mean(cov[-WIN:]):.4f} < {COV_DROP} x its mean over steps 1 - {WIN} {_mean(cov[:WIN]):.4f}"
    if len(na) > WIN and _mean(na[-WIN:]) > NACT_RISE * _mean(na[:WIN]):
        return f"the token set blew up: the active cells per frame (uncapped) over the last {WIN} steps {_mean(na[-WIN:]):.1f} > {NACT_RISE} x its mean over steps 1 - {WIN} {_mean(na[:WIN]):.1f}"
    lo, hi = H_LATE
    if done >= hi and _mean(h[lo - 1:hi]) > H_RISE * _mean(h[:WIN]):
        return f"the head's loss over steps {lo} - {hi} {_mean(h[lo - 1:hi]):.4f} > {H_RISE} x its mean over steps 1 - {WIN} {_mean(h[:WIN]):.4f}"
    return None


def nn_min(pos, valid):
    """the smallest distance in m between two different valid tokens of one frame (pos (B, K, 2), valid (B, K)): exact differences in float64, no matmul (near-duplicates); NaN without such a pair"""
    with torch.no_grad():
        v = valid.bool()
        ok = v[:, :, None] & v[:, None] & ~torch.eye(v.shape[1], dtype=torch.bool, device=v.device)
        p = pos.detach().double()
        d2 = (p[:, :, None] - p[:, None]).square().sum(-1)
        return float(d2[ok].min().sqrt()) if bool(ok.any()) else float("nan")


def adam_update(g, state, group):
    """the step that AdamW makes for one tensor with the gradient g, from its state and group BEFORE the step (bias-corrected moments as torch's AdamW; the decoupled weight decay is left out: it does not depend on the gradient)"""
    assert not group.get("amsgrad") and not group.get("maximize"), "plain AdamW only"
    b1, b2 = group["betas"]
    t = float(state["step"]) + 1 if state else 1.0
    m = (1 - b1) * g + (b1 * state["exp_avg"] if state else 0.0)
    v = (1 - b2) * g * g + (b2 * state["exp_avg_sq"] if state else 0.0)
    return group["lr"] / (1 - b1 ** t) * m / (v.sqrt() / math.sqrt(1 - b2 ** t) + group["eps"])


def shares(params, gb, opt):
    """the card's secondary numbers on one step: params = the node head (.grad = the step's full gradient, before the clip), gb = the gradient of b_weight x B''s loss on them (autograd.grad on the same step), g_head = .grad - gb, v_hat = AdamW's
    bias-corrected second moment of the state before the step -> dict(ushare = |g_head / (sqrt(v_hat) + eps)| / (that + |gb / (sqrt(v_hat) + eps)|), NaN without a state (the first step, after a roll-back); gshare = |g_head| / (|g_head| + |gb|))"""
    groups = {id(p): g for g in opt.param_groups for p in g["params"]}
    hp, bp, hr, br, have = [], [], [], [], True
    for p, b in zip(params, gb):
        if p.grad is None:
            continue
        b = torch.zeros_like(p) if b is None else b
        hd = p.grad - b
        st, grp = opt.state.get(p, {}), groups[id(p)]
        hr.append(hd)
        br.append(b)
        if st:
            d = (st["exp_avg_sq"] / (1 - grp["betas"][1] ** float(st["step"]))).sqrt() + grp["eps"]
            hp.append(hd / d)
            bp.append(b / d)
        else:
            have = False
    u_h, u_b, g_h, g_b = _norm(hp), _norm(bp), _norm(hr), _norm(br)
    return dict(ushare=u_h / (u_h + u_b) if have and u_h + u_b > 0 else float("nan"), gshare=g_h / (g_h + g_b) if g_h + g_b > 0 else float("nan"))


class StepHook:
    """what `trainer.fit` does not offer, run by torch's global optimiser pre-step hook (after fit's backward and its clip, which is off here, right before AdamW's step), on fit's optimiser only (the one that holds the head): at the first update of a process no
    tensor of B' outside the trained node head holds a gradient (z: none of B'); on a share step (s4) the shares (`shares`, on the gradient before the clip); THE CLIP PER GROUP (the head, the node head: CLIP each); on a logged step the clip factors and the head's
    update norm into the step's info (kill when it is not finite and > 0). `cur` = what `loss_fn` hands over for this step (i, info, gb, logged)"""

    def __init__(self, run, joint):
        self.run, self.cur, self.first = run, None, True
        self.heads, self.nodes, self.pyrs = list(joint.head.parameters()), joint.src.node_params(), joint.src.pyr_params()
        self.own = {id(p) for p in self.heads}
        trained = {id(p) for p in self.nodes + self.pyrs}
        self.frozen = [(n, p) for n, p in joint.src.named_parameters() if id(p) not in trained]

    def __call__(self, opt, args, kwargs):
        if not any(id(p) in self.own for g in opt.param_groups for p in g["params"]):
            return                                                                              # another optimiser
        cur, self.cur = self.cur or {}, None
        info = cur.get("info", {})
        if self.first:
            held = [n for n, p in self.frozen if p.grad is not None and bool(p.grad.ne(0).any())]
            if held:
                raise Kill(f"B''s tensors outside the trained node head hold a gradient: {held[:5]} ({len(held)})")
            self.first = False
        if cur.get("gb") is not None:
            sh = shares(self.nodes, cur["gb"], opt)
            info.update(sh)
            self.run.log(f"step {cur['i']}: update share of the head's loss on B''s node head {sh['ushare']:.3f} (one step, preconditioned by AdamW's v_hat), gradient-norm share {sh['gshare']:.3f}")
        nh, nn_, np_ = (torch.nn.utils.clip_grad_norm_([p for p in ps if p.grad is not None], CLIP) for ps in (self.heads, self.nodes, self.pyrs))
        if cur.get("logged"):
            groups = {id(p): g for g in opt.param_groups for p in g["params"]}
            upd = lambda ps: _norm([adam_update(p.grad, opt.state.get(p, {}), groups[id(p)]) for p in ps if p.grad is not None])  # noqa: E731
            u = upd(self.heads)
            info.update(clip_head=min(1.0, CLIP / (float(nh) + 1e-6)), clip_nodes=min(1.0, CLIP / (float(nn_) + 1e-6)), upd_head=u)
            if self.pyrs:                                                                       # E393: the pyramid's group (its gradient norm before the clip, its share, its clip factor, the update norms of both B' groups)
                info.update(gn_head=float(nh), gn_nodes=float(nn_), gn_pyr=float(np_), pshare=float(np_) / max(float(nh) + float(nn_) + float(np_), 1e-12), clip_pyr=min(1.0, CLIP / (float(np_) + 1e-6)),
                            upd_nodes=upd(self.nodes), upd_pyr=upd(self.pyrs))
            if not (math.isfinite(u) and u > 0):
                raise Kill(f"step {cur.get('i')}: the head's update norm is {u} (not learning)")


class Ran:
    """the head's output of this step as `jtoken_train.make_loss` calls the head, m(b): the head ran on the live tokens before the labels existed (the overlap)"""

    def __init__(self, head, out):
        self.head, self.out = head, out

    def __call__(self, b):
        return self.out

    def parameters(self):
        return self.head.parameters()

    @property
    def _step_py(self):
        return self.head._step_py


def flag(run, key, step, text):
    """a broken pair with the twin arm made visible: the log line and the step appended to status.json[key]"""
    run.log(text)
    run.status(**{key: list(run._status.get(key, [])) + [step]})


def make_loss(run, joint, args, hook):
    """the loss of one step (module docstring) with the kill rules and the log; the head's loss and its log lines (first loss, clock, peak memory) are `jtoken_train.make_loss`'s"""
    base = JKT.make_loss(run, none_weight=args["none_weight"], pos_loss=args["pos_loss"], wire_w=args["wire_weight"])
    tn, live, w, steps = bool(args["train_nodes"]), bool(args["live_inputs"]), args["b_weight"], args["steps"]
    b_rise = tn and w > 0                                                                       # B''s node loss is in the loss: its rise is a kill (E393 n / e: logged only)
    nodes, pyrs = joint.src.node_params(), joint.src.pyr_params()
    win, st = collections.deque(maxlen=LOG_EVERY), dict(last=None, checked=False)               # win: the label numbers of the last 100 steps of the process

    def loss_fn(m, b):
        i = len(m.rec["b"]) + 1                                                                 # this step, the skipped ones counted (= fit's step after it)
        if st["last"] is None and i > 1:
            flag(run, "resumed_at", i - 1, f"RESUMED at step {i - 1}: a new process draws a new batch order, the pair with the twin arm is broken from here")
        elif st["last"] is not None and i <= st["last"]:
            flag(run, "rollbacks_at", i - 1, f"ROLL-BACK to step {i - 1} (fit's guard: weights and Adam moments of the checkpoint, lr x 0.7): the pair with the twin arm is broken from here")
        st["last"], hook.cur = i, None
        if m.head.live_inputs != live:
            raise Kill(f"the head's live_inputs is {m.head.live_inputs}, arm {arm(tn, live)} needs {live}")
        no, tk = m.src.tokens(b["x"])
        out = m.head(LLT.head_inputs(tk))                                                       # queued before the labels
        n_act = (no["e"].detach().float().flatten(1).sigmoid() > LLT.LC.P_ACTIVE).sum(1).float().mean()     # the active cells per frame before the top-K cut (read back after the label step's sync)
        if i <= FRAMES_LOGGED:
            run.log(f"step {i} frames {list(b['token'])}")
        try:
            lb = m.src.labelled(tk, b["table"], b["token"], b["row"], token_drop=0.0, gate=False)
        except LLT.LabelOverflow as e:
            for k in ("b", "h", "cov", "nact"):
                m.rec[k].append(float("nan"))
            m.rec["overflow"].append(i)
            run.log(f"step {i}: LabelOverflow ({e}): skipped, {len(m.rec['overflow'])} so far")
            why = verdict(m.rec, b_rise)
            if why:
                raise Kill(why)
            return torch.full((), float("nan")), dict(overflow=float(len(m.rec["overflow"])))
        loss_h, info = base(Ran(m.head, out), lb)
        with torch.autocast(device_type=no["e"].device.type, enabled=False), torch.set_grad_enabled(tn):
            loss_b, info_b = LLT.b_loss(no, b["targets"])
        assert loss_b.dtype == torch.float32, f"B''s node loss in {loss_b.dtype}, not float32"
        if not (math.isfinite(float(loss_h)) and math.isfinite(float(loss_b))):
            raise Kill(f"step {i}: a loss is not finite (the head's {float(loss_h)}, B''s node loss {float(loss_b)})")
        loss = loss_h + w * loss_b if b_rise else loss_h                                        # (b_weight 0: B''s loss stays out of the backward, a non-finite gradient there cannot reach the step)
        gb = None
        if tn and not st["checked"]:                                                            # the first step of the process: does the head's loss reach the node head (s4) / not (b)?
            n_h = _norm(torch.autograd.grad(loss_h, nodes, retain_graph=True, allow_unused=True))
            if live and not n_h > 0:
                raise Kill(f"refused at the start: the head's loss gives B''s node head no gradient (norm {n_h}): the head is not live")
            if not live and n_h != 0:
                raise Kill(f"arm b: the head's loss reaches B''s node head (gradient norm {n_h:.4g})")
            run.log(f"step {i}: the head's loss gives B''s node head a gradient of norm {n_h:.4g} ({'> 0: s4' if live else '0: arm b'})")
            if pyrs:                                                                            # E393: the head's loss reaches the pyramid (its norm finite and > 0); the tensors it does not reach are listed (lines05, by design)
                gp = torch.autograd.grad(loss_h, pyrs, retain_graph=True, allow_unused=True)
                n_p = _norm(gp)
                if not (math.isfinite(n_p) and n_p > 0):
                    raise Kill(f"refused at the start: the head's loss gives B''s pyramid no gradient (norm {n_p})")
                pn = [n_ for n_, p in joint.src.model.pyr.named_parameters() if p.requires_grad]
                zero = [n_ for n_, g_ in zip(pn, gp) if g_ is None or not bool(g_.ne(0).any())]
                run.log(f"step {i}: the head's loss gives B''s pyramid a gradient of norm {n_p:.4g}; pyramid tensors without it: {zero}")
            st["checked"] = True
        if tn and live and (i % SHARE_EVERY == 0 or i == steps):
            gb = torch.autograd.grad(w * loss_b, nodes, retain_graph=True, allow_unused=True)
        logged = i % LOG_EVERY == 0 or i == steps
        hook.cur = dict(i=i, info=info, gb=gb, logged=logged)
        if logged:                                                                              # the numeric hazards of moving tokens: the closest pair, the head's pull on the positions (set by fit's backward)
            info.update(nn_min=nn_min(tk["pos"], tk["valid"]), pos_grad_max=float("nan"))
            if live and tk["pos"].requires_grad:
                tk["pos"].register_hook(lambda g_: info.__setitem__("pos_grad_max", float(g_.abs().max())))
        m.rec["b"].append(float(loss_b))
        m.rec["h"].append(float(loss_h))
        lab = lb["lab"]
        win.append((info["pos_pairs"], info["gt_links"], sum(float(x["n_gt_assigned"]) for x in lab), sum(float(x["n_gt"]) for x in lab), sum(float(x["n_valid"]) for x in lab), len(lab), lb["label_s"]))
        m.rec["cov"].append(win[-1][2] / win[-1][3] if win[-1][3] > 0 else float("nan"))
        m.rec["nact"].append(float(n_act))
        s = [sum(c) for c in zip(*win)]
        info.update(head=float(loss_h), b_loss=float(loss_b), **{"b_" + k: float(v) for k, v in info_b.items()}, b_start=_mean(m.rec["b"][:WIN]) if i >= WIN else float("nan"), b_mean=_mean(m.rec["b"][-WIN:]),
                    h_start=_mean(m.rec["h"][:WIN]) if i >= WIN else float("nan"), overflow=float(len(m.rec["overflow"])),
                    pair_cov=s[0] / max(s[1], 1.0), gt_cov=s[2] / max(s[3], 1.0), n_valid=s[4] / s[5], label_s=s[6] / len(win),
                    cov_start=_mean(m.rec["cov"][:WIN]) if i >= WIN else float("nan"), cov_mean=_mean(m.rec["cov"][-WIN:]), nact_start=_mean(m.rec["nact"][:WIN]) if i >= WIN else float("nan"), nact_mean=_mean(m.rec["nact"][-WIN:]))
        why = verdict(m.rec, b_rise)
        if why:
            raise Kill(why)
        return loss, info
    return loss_fn


# ------------------------------------------------------------------------------------------------------------------------------ model.pt
def load_b(run, dev, runs=None, name="model.pt"):
    """-> B' as a jlive run trained it: a `QGraph` in eval mode, built from the B' run's args as `qpredict.load_model` builds a run, with the run's b_state"""
    ck = torch.load(os.path.join(runs or T.RUNS, run, name), map_location="cpu", weights_only=False)
    a = ck["b_args"]
    m = QG.QGraph(width=int(a.get("width", 48)), soft_nms=a.get("soft_nms"), ends=bool(a.get("ends")))
    m.load_state_dict(ck["b_state"])
    return m.to(dev).eval()


def export(run, joint, args, b_args, start):
    """model.pt (module docstring) written, read back and checked: the head and B' as trained, bit for bit, the head's live_inputs the arm's; B''s tensors outside the node head (z: all of them) as at the start (`start`: B''s state when it was loaded).
    Any failure (a check, the write, the strict load) renames fit's .done to .done.FALSE and sets status.json `state` failed (fit wrote done before the export)"""
    p = run.path("model.pt")
    same = lambda x, y: list(x) == list(y) and all(torch.equal(x[k].cpu(), y[k].cpu()) for k in x)  # noqa: E731
    try:
        torch.save(dict(state=joint.head.state_dict(), args=args, steps=args["steps"], b_state=joint.src.model.state_dict(), b_args=b_args, rec=joint.get_extra_state()), p + ".tmp")
        os.replace(p + ".tmp", p)
        runs = os.path.dirname(run.dir)
        head, _ = JKT.load_model(run.name, "cpu", runs)
        b = load_b(run.name, "cpu", runs).state_dict()
        assert same(head.state_dict(), joint.head.state_dict()) and head.live_inputs == bool(args["live_inputs"]), "model.pt: the head read back is not the trained head"
        assert same(b, joint.src.model.state_dict()), "model.pt: B' read back is not the trained B'"
        moved = [k for k in b if not torch.equal(b[k].cpu(), start[k].cpu()) and not (args["train_nodes"] and k.startswith("nodes.")) and not (args.get("train_pyr") and k.startswith("pyr."))]
        assert not moved, f"model.pt: B''s tensors that must stay as at the start moved: {moved[:5]}"
    except Exception as e:
        if os.path.exists(run.path(".done")):
            os.replace(run.path(".done"), run.path(".done.FALSE"))
        run.status(state="failed", error=repr(e))
        raise


# ------------------------------------------------------------------------------------------------------------------------------ the loop
def train(a, runs=T.RUNS, ckpt_every=500):
    """the loop of the parsed arguments `a` (no GPU check: `main` has it)"""
    args = resolve(a, runs)
    check_start(args, os.path.join(runs, a.name))
    joint, b_args = build(args, runs)
    start = {k: v.clone() for k, v in joint.src.model.state_dict().items()}
    loader = batches(a.seed, a.bs, a.workers, args["b_run"], args["ident"])
    run = T.Run(a.name, root=runs)
    if a.train_pyr:                                                                             # E393: the pyramid in its own AdamW group at pyr_lr_mult x the schedule's lr (trainer.fit's lr_mult)
        params = [dict(params=list(joint.head.parameters()) + joint.src.node_params()), dict(params=joint.src.pyr_params(), lr_mult=a.pyr_lr_mult)]
    else:
        params = list(joint.head.parameters()) + joint.src.trainable()
    tn, live = bool(a.train_nodes), bool(a.live_inputs)
    n_par = lambda ps: sum(p.numel() for p in ps)  # noqa: E731
    run.log(f"jlive arm {arm(tn, live)}{' + pyramid' if a.train_pyr else ''}: head {a.head_run} (sha256 {args['head_sha256'][:12]}, head step {joint.head._step_py}), B' {args['b_run']} (sha256 {args['b_sha256'][:12]})"
            + (f", its state from {a.b_state_from} (b_state, sha256 {args['b_state_sha256'][:12]})" if a.b_state_from else "") + f", live_inputs {joint.head.live_inputs}, train_nodes {a.train_nodes}, train_pyr {a.train_pyr}; "
            f"trained: the head ({n_par(joint.head.parameters())} parameters)" + (f" + B''s node head ({n_par(joint.src.node_params())} parameters, {len(joint.src.node_params())} tensors)" if tn else ", B' frozen")
            + (f" + B''s pyramid ({n_par(joint.src.pyr_params())} parameters, {len(joint.src.pyr_params())} tensors, lr x {a.pyr_lr_mult}, its own clip group)" if a.train_pyr else "")
            + f"; loss: the head's (none weight {args['none_weight']}, pos loss {args['pos_loss']})" + (f" + {a.b_weight} x B''s node loss" if tn else ", B''s node loss logged only")
            + f"; {a.steps} steps, bs {a.bs}, lr {a.lr} (cosine, 5 % warm-up, every group), clip {CLIP} per group (fit's joint clip off), head seed {args['head_seed']} (data order), B' seed {a.seed}, K {K}, token drop 0, label gate off")
    hook = StepHook(run, joint)
    handle = register_optimizer_step_pre_hook(hook)
    try:
        T.fit(joint, make_loss(run, joint, args, hook), run, steps=a.steps, loader=loader, lr=a.lr, sched="cosine", warmup=0.05, clip=0, ckpt_every=ckpt_every, seed=args["head_seed"], args=args, params=params,
              log_every=LOG_EVERY, device="cuda" if torch.cuda.is_available() else "cpu", config=dict(design="jlive", clip_per_group=CLIP))
    except Kill as e:                                                                           # never resumed: the reason in KILLED (`check_start`) and in status.json
        with open(run.path("KILLED"), "w", encoding="utf-8") as fh:
            fh.write(str(e) + "\n")
        run.status(killed=str(e))
        raise
    finally:
        handle.remove()
    export(run, joint, args, b_args, start)
    run.log(f"done {a.name}: model.pt holds the head and B' (b_state), read back and checked")


def main(argv=None, runs=T.RUNS):
    a = parse_args(argv)
    assert torch.cuda.is_available(), "no GPU visible: a run of the card trains on the GPU (bf16)"
    train(a, runs)


if __name__ == "__main__":
    main()
