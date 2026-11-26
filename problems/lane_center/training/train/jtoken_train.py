"""Trainer of the E365 token head on the predicted-node cache of the frozen lane head B' (stage S1 of the token champion); start it through `autoresearch/tools/run.py`.

    wsl bash tools/env/_pyr.sh jtoken_train.py --name e365_s1_s0 --seed 0 --ident e292_a --steps 12000 --bs 8 --lr 1e-3 --workers 4 [--cache <cache root>] [--K 384] [--distill-from runs/A,runs/B [--distill-weight 1.0]] [--smoke]

The model is `net/model/jtoken.JTokenHead` (the whole trainable path: B' is cached, nothing of it runs), the loss `jtoken.jtoken_loss`. Batches come from `net/prep/lane_ptok_data.loader("train", seed, bs, workers, ident=, K=,
cache=)` (imported only when the loop starts): `--cache` = the cache ROOT (default `runs/e365_cache`), `--ident` = the frozen head whose tokens are cached (`e292_a` = B', `e273_p` = plain B), `--K` only when the root
holds more than one K for the seed. `--smoke` = 20 steps on synthetic batches (`jtoken.synthetic_batch`, K 384, no cache, no loader): the plumbing test and the memory check.
The loop, the run-directory contract, the checkpoint every 500 steps with resume, the roll-back guard and the `clock` line are those of `trainer.fit`; AdamW lr 1e-3, 5 % warm-up, cosine to 0.
The architecture is the champion's, a constant (`ARCH`: d 64, heads 4, layers 3, ffn 128; `arch`, `build_head`; `load_model` rebuilds it).
`--none-weight W` (default 1 = the loss of before) multiplies the loss of the sources whose GT successor set is empty (target: the none slot) and the type cross-entropy of the none class (`jtoken.jtoken_loss`).
`--pos-loss max` (E376; default sum = the loss of before): a source with GT successors is trained on -log max over its group instead of -log of its sum (one winner among the duplicates; a fork's group holds both branches).
`live_inputs` (S4, E372: the head keeps the graph of its inputs,
`jtoken.JTokenHead(live_inputs=)`) has no flag here (the cached inputs carry no graph): `build_head` reads it from a run's args (absent = False), so a joint run's model.pt rebuilds with it; not in `arch` and not in the
recipe. Options removed in D117 (`REMOVED`: the c-rule `prior` / `teacher` of E369 / E370, E370's `init`, the sizes d / heads / layers / ffn, `token_drop`, E377's `none_weight_near`): a run's args that hold their default (or
nothing) load and resume as before, another value is refused (`check_removed`). `--head-seed N` (default: --seed):
the head's initialisation and the data order use N, `--seed` keeps choosing the B' cache (args['seed'] stays the cache seed, so the read-outs open the same cache); in the recipe.
`--distill-from runs/A,runs/B` (E379 card v2, ensemble distillation; unset = the loss of before): the heads of these runs (`load_teachers`: eval, frozen, their own step counter) run on the step's batch in no_grad, float32 with
autocast OFF (the student stays in the trainer's autocast), and the loss is the GT loss + `--distill-weight` W (default 1) x `jtoken.distill_loss` from step 1 (gt = the GT loss alone, kd, kd_s, kd_l and their means gt_mean / kd_mean over the last LOG_WINDOW steps in the info of every step; at the first batch of
a process the parts, the teachers' own GT loss, the KL of their mean in float32 against bf16 autocast and the frame tokens in the log). HARD start checks (also at a resume): a teacher learned on the student's tokens
(cache seed, ident, K, cache root), its step counter is its run's steps, the sha256 of its model.pt is in the recipe (`distill_sha256`: a changed file is refused at a resume), and the MEAN of its own GT loss over the run's first START_BATCHES
batches (`first_batches`) lies in TEACHER_GT.
`--noise-pos SP` / `--noise-e SE` (E381 card v2, D126; default 0 = the loss of before bit for bit): per training step the head's INPUTS are perturbed (`add_noise`: valid tokens, pos += N(0, SP^2) per axis, e and e_raw += one
N(0, SE^2) draw, an own generator keyed on (head seed, the head's step counter): a resume repeats the draw, the data order is untouched); the label tensors are the clean tokens', but the gate and the pair geometry come from the
perturbed positions (a GT link can leave the gate: `gt_outside` in the clock line); the first-loss line holds the CLEAN loss of the first batch and its frame tokens, every clock line the mean displacement and mean |noise e|;
the read-outs (`jtoken_eval.py`) take no noise. In the recipe.
`--rounds R` (E392 card v3, 0 / 1 / 2; default 0 = the champion bit for bit) / `--msg real|uniform|none|self` (uniform = the content null; self = the capacity null: the round modules on each token's own features, no neighbour read, any --wire-weight; none = deep supervision without exchange: no round module, the early reads exist in training mode only, so the
trained head is the champion network at inference; needs --rounds >= 1 and --wire-weight > 0) / `--wire-weight W` (default 0): the head's message rounds after layer 1 / 2 (`jtoken.JTokenHead`) and the weight of the likelihood of
their early wiring reads (`jtoken.jtoken_loss(wire_w=)`); in the recipe and in model.pt's args, so `build_head` (the trainer, `load_model`, the stage-2 trainer) rebuilds the head with them; absent in an old run's args = the default.
`--pos-head` / `--curve-w W` / `--node-w W` (E395 card, default off = the champion bit for bit): the position head of the token head (`jtoken.JTokenHead(pos_head=)`, a normal offset per token, zero start) and the weight of its loss, the Catmull-Rom curve loss
(`--curve-w`) or the node loss (`--node-w`, the control; not both), `jtoken.jtoken_loss(curve_w=, node_w=)` on the GT geometry of `net/prep/lane_gtgeom_data` (the loader is built with geom=True; `--smoke`: the synthetic toy geometry); both weights need --pos-head and are refused with --distill-from;
the log row of such a run also holds `pos_mean` (the mean of the trained position loss over the last LOG_WINDOW calls) and `pos_n` (the calls in that mean), the reading of the card's no-learning kill from addendum A1 on (`curve` / `node` are one batch);
in the recipe and in model.pt's args (absent in an old run's args = off), so `build_head` rebuilds the head with it. The stage-2 trainer (`jlive_train`) refuses a head run that has a position head.
The key-weight switch of the head counts the head's own training forwards (a buffer inside the checkpoint), so a resume or a roll-back keeps it right. The scored model is the final one (nothing is chosen on VAL).
A folder that holds another recipe is refused; the same recipe is resumed. `main` needs a GPU unless `--smoke`; `train(args, runs, ckpt_every, smoke_k)` is the loop without that check, and `ckpt_every` / `smoke_k`
are TEST-ONLY arguments (a run uses 500 / 384).
"""
import argparse
import collections
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub_ in ("training", "net/prep", "net/model", "net/loss", "training/data", "training/data/prep", "evaluation"):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import bflc_paths as BP  # noqa: E402  (where code, data and runs are)
import jtoken as JK  # noqa: E402
import loss_jtoken as LJK  # noqa: E402
import trainer as T  # noqa: E402

RECIPE = ("seed", "cache", "ident", "K", "bs", "steps", "lr", "none_weight", "pos_loss", "head_seed",
          "distill_from", "distill_weight", "distill_sha256", "noise_pos", "noise_e", "rounds", "wire_weight", "msg", "pos_head", "curve_w", "node_w")
ARCH_DEFAULTS = dict(none_weight=1.0, pos_loss="sum", head_seed=None, live_inputs=False,
                     distill_from=None, distill_weight=1.0, distill_sha256=None, noise_pos=0.0, noise_e=0.0,
                     rounds=0, wire_weight=0.0, msg="real", pos_head=False, curve_w=0.0, node_w=0.0)                 # what a run of before these options means (the head of the first build)
ARCH = dict(d=64, heads=4, layers=3, ffn=128)                               # the champion's architecture (the head of the first build), a constant since D117
REMOVED = dict(prior="none", teacher="none", init=None, token_drop=0.0, none_weight_near=None, **ARCH)     # D117: options that no longer exist -> the value a stored run must hold (or nothing): today's head and loss are then that run's
SMOKE_K = 384                                                               # tokens per synthetic frame of the smoke run: the memory of the real K
TEACHER_GT, START_BATCHES = (0.30, 0.50), 20                                # E379 card v2: the band of a teacher's own GT loss (none weight 0.25: 0.37 on the 291-frame TRAIN sample), the MEAN over the run's first 20
                                                                            # batches (a single batch: SD ~0.08, outside the band 24 - 26 % of the time)
LOG_WINDOW = 100                                                            # `trainer.fit`'s log_every: the window of gt_mean / kd_mean (one log row per window when no step is skipped)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", help="the cache ROOT (default runs/e365_cache; the loader finds the folder of the seed, the ident, K and the GT signature in it)")
    ap.add_argument("--ident", default="e292_a", help="the frozen head of the cached tokens: e292_a = B', e273_p = plain B")
    ap.add_argument("--K", type=int, help="tokens per frame, only when the root holds more than one K for the seed")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--none-weight", type=float, default=ARCH_DEFAULTS["none_weight"], dest="none_weight", help="W: multiplies the loss of the sources with an empty G(i) (target: none) and the type cross-entropy of the none class (1 = off)")
    ap.add_argument("--pos-loss", choices=LJK.POS_LOSS, default=ARCH_DEFAULTS["pos_loss"], dest="pos_loss", help="E376: the likelihood of a source with GT successors, -log sum over its group (sum) or -log max (max)")
    ap.add_argument("--head-seed", type=int, default=None, dest="head_seed", help="the seed of the head's initialisation and the data order (default: --seed); --seed keeps choosing the B' cache")
    ap.add_argument("--distill-from", default=ARCH_DEFAULTS["distill_from"], dest="distill_from", help="E379: comma list of run folders (relative to the problem folder) whose frozen heads are the teachers of the KD term")
    ap.add_argument("--distill-weight", type=float, default=ARCH_DEFAULTS["distill_weight"], dest="distill_weight", help="E379: W, the loss = the GT loss + W x the KD term (only with --distill-from)")
    ap.add_argument("--noise-pos", type=float, default=ARCH_DEFAULTS["noise_pos"], dest="noise_pos", help="E381: per training step pos += N(0, SP^2) per axis on the valid tokens (m; 0 = off)")
    ap.add_argument("--noise-e", type=float, default=ARCH_DEFAULTS["noise_e"], dest="noise_e", help="E381: per training step e and e_raw += the same N(0, SE^2) draw on the valid tokens (logit; 0 = off)")
    ap.add_argument("--rounds", type=int, choices=(0, 1, 2), default=ARCH_DEFAULTS["rounds"], help="E392: message rounds after layer 1 / layer 2 (0 = the champion)")
    ap.add_argument("--wire-weight", type=float, default=ARCH_DEFAULTS["wire_weight"], dest="wire_weight", help="E392: W, + W x the likelihood of every early wiring read (0 = none; needs --rounds)")
    ap.add_argument("--msg", choices=JK.MSG, default=ARCH_DEFAULTS["msg"], help="E392: the weights of a round: the early read's probabilities (real), the content null (uniform) or no exchange (none: deep supervision only, needs --wire-weight > 0) or the round block on each token's own features (self: the capacity null); not real needs --rounds")
    ap.add_argument("--pos-head", action="store_true", dest="pos_head", help="E395: the position head (a normal offset per token, zero start); needs --curve-w or --node-w to learn")
    ap.add_argument("--curve-w", type=float, default=ARCH_DEFAULTS["curve_w"], dest="curve_w", help="E395: W, + W x the Catmull-Rom curve loss of the refined nodes (0 = off; needs --pos-head)")
    ap.add_argument("--node-w", type=float, default=ARCH_DEFAULTS["node_w"], dest="node_w", help="E395: W, + W x the node loss of the refined nodes (the control of the curve loss; 0 = off; needs --pos-head; not with --curve-w)")
    ap.add_argument("--smoke", action="store_true", help="20 steps on synthetic batches")
    a = ap.parse_args(argv)
    check_options(vars(a))
    return a


NOISY = ("pos", "e", "e_raw")                                               # E381: the batch keys `add_noise` changes (every other key, the labels among them, is passed on as the same object)


def add_noise(b, sp, se, seed, step):
    """E381 (card v2, D126): the head's inputs of one training step perturbed, the labels untouched -> (batch, (mean displacement of the valid tokens in m, mean |noise| on e)) or (b itself, None) when both are 0.
    pos += N(0, sp^2) per axis, e and e_raw += the SAME N(0, se^2) draw, valid tokens only; drawn on the CPU from an own generator seeded by (seed, step) (the head seed and the head's own step counter, restored at a resume:
    a resumed run repeats the draw; the global random stream, i.e. the data order, is not touched). The head computes its gate and pair geometry from the perturbed positions (so a GT link can leave the gate)"""
    if not sp and not se:
        return b, None
    g = torch.Generator().manual_seed(int(seed) * 1_000_003 + int(step))
    valid = b["valid"].bool().cpu()
    vf = valid.to(torch.float32)
    npos = torch.randn(b["pos"].shape, generator=g) * sp * vf[..., None]
    ne = torch.randn(b["e"].shape, generator=g) * se * vf
    out = dict(b)
    out["pos"] = b["pos"] + npos.to(b["pos"].device, b["pos"].dtype)
    out["e"] = b["e"] + ne.to(b["e"].device, b["e"].dtype)
    out["e_raw"] = b["e_raw"] + ne.to(b["e_raw"].device, b["e_raw"].dtype)
    return out, (float(npos.norm(dim=-1)[valid].mean()), float(ne.abs()[valid].mean()))


def check_options(args):
    """before anything is built or loaded: the teachers of --distill-from have a model.pt and the KD weight is >= 0 (not given without teachers); the early-read weight is >= 0 and neither it nor --msg uniform / none is given without --rounds;
    --msg none (the champion network with extra reads, trained only through their loss) needs --wire-weight > 0; the E395 weights --curve-w / --node-w are >= 0, need --pos-head, are not given together and not with --distill-from"""
    if args.get("distill_from") is not None:
        ps = distill_paths(args["distill_from"])
        assert ps and all(os.path.isfile(os.path.join(p, "model.pt")) for p in ps), f"--distill-from {args['distill_from']}: run folders with a model.pt"
        assert args["distill_weight"] >= 0.0, f"--distill-weight {args['distill_weight']}: >= 0"
    else:
        assert args.get("distill_weight", 1.0) == 1.0, f"--distill-weight {args['distill_weight']} without --distill-from"
    assert args.get("noise_pos", 0.0) >= 0.0 and args.get("noise_e", 0.0) >= 0.0, f"--noise-pos {args.get('noise_pos')} / --noise-e {args.get('noise_e')}: >= 0"
    assert args.get("wire_weight", 0.0) >= 0.0, f"--wire-weight {args.get('wire_weight')}: >= 0"
    assert args.get("rounds", 0) > 0 or (args.get("wire_weight", 0.0) == 0.0 and args.get("msg", "real") == "real"), f"--wire-weight {args.get('wire_weight')} / --msg {args.get('msg')} without --rounds"
    assert args.get("msg", "real") != "none" or args.get("wire_weight", 0.0) > 0.0, "--msg none without --wire-weight > 0 (it would train the champion with extra reads that no loss sees)"
    assert not ((args.get("noise_pos", 0.0) > 0 or args.get("noise_e", 0.0) > 0) and args.get("msg", "real") in ("none", "self") and args.get("wire_weight", 0.0) > 0.0), "--noise-pos / --noise-e with --msg none / self and --wire-weight > 0: the clean first-batch loss runs in eval mode, which has no early read"
    cw, nw = args.get("curve_w", 0.0), args.get("node_w", 0.0)
    assert cw >= 0.0 and nw >= 0.0, f"--curve-w {cw} / --node-w {nw}: >= 0"
    assert args.get("pos_head", False) or not (cw or nw), f"--curve-w {cw} / --node-w {nw} without --pos-head (no position to refine)"
    assert not (cw and nw), f"--curve-w {cw} and --node-w {nw} together: one loss per arm (the node loss is the control of the curve loss)"
    assert args.get("distill_from") is None or not (cw or nw), f"--distill-from together with --curve-w {cw} / --node-w {nw}: not supported (the teachers' heads have no position head: their own GT loss would need the curve term)"


def _abs(p):
    return BP.resolve(p)


def distill_paths(spec):
    """--distill-from 'runs/A,runs/B' -> the teachers' run folders (relative to the problem folder, or absolute)"""
    return [os.path.normpath(_abs(p.strip())) for p in str(spec or "").split(",") if p.strip()]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def check_removed(args, where):
    """a run's args that hold an option of REMOVED at another value than its default are refused: that head or loss no longer exists (the tag archive/jtoken-before-simplify-2026-11-17 keeps it)"""
    held = {k: args[k] for k, v in REMOVED.items() if args.get(k) not in (None, v)}
    assert not held, f"{where}: {held}: options removed in D117 (the tag archive/jtoken-before-simplify-2026-11-17 keeps them)"


def check_start(args, folder):
    """before anything is written: a folder that already holds a run is only RESUMED (the same recipe), never another run's; a run of a removed option is refused (`check_removed`)"""
    held = [f for f in ("config.json", "ckpt.pt", "model.pt") if os.path.exists(os.path.join(folder, f))]
    if held:
        c = json.load(open(os.path.join(folder, "config.json"))) if "config.json" in held else {}
        check_removed(c.get("args") or {}, folder)
        old = {k: (c.get("args") or {}).get(k, ARCH_DEFAULTS.get(k)) for k in RECIPE}                    # a run of before an option has its default
        assert c.get("design") == "jtoken" and old == {k: args.get(k, ARCH_DEFAULTS.get(k)) for k in RECIPE}, f"{folder} holds {held} of another run ({c.get('design')}, {old}): use another --name"


def to_device(b, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def build_head(args):
    """the head of a run's args (live_inputs, rounds, msg, pos_head: absent = the default; the architecture `arch`): the one place that builds it, for the trainer, `load_model` and the stage-2 trainer"""
    d, heads, layers, ffn = arch(args)
    return JK.JTokenHead(d=d, heads=heads, layers=layers, ffn=ffn, live_inputs=bool(args.get("live_inputs", ARCH_DEFAULTS["live_inputs"])),
                         rounds=int(args.get("rounds", ARCH_DEFAULTS["rounds"])), msg=args.get("msg", ARCH_DEFAULTS["msg"]), pos_head=bool(args.get("pos_head", ARCH_DEFAULTS["pos_head"])))


def arch(args):
    """(d, heads, layers, ffn) = the constants ARCH; the args of a removed option at another value are refused (`check_removed`)"""
    check_removed(args, "the run's args")
    return ARCH["d"], ARCH["heads"], ARCH["layers"], ARCH["ffn"]


def make_loss(run, none_weight=1.0, pos_loss="sum", teachers=(), distill_weight=1.0, noise_pos=0.0, noise_e=0.0, noise_seed=0, wire_w=0.0, curve_w=0.0, node_w=0.0):
    """`jtoken.jtoken_loss` (+ wire_w x the early reads, E392; + curve_w x the curve loss or node_w x the node loss of the position head, E395: their info in the clock line; the teachers' own GT loss without the early reads) + what the log needs: every step the peak GPU memory; at call 50 and every 100 calls a clock line (seconds per step since the last line); teachers (E379, the frozen heads of `load_teachers`): `teacher_forward` on the
    same batch, + distill_weight x `jtoken.distill_loss` from the first step, its parts in the info; at the first call the parts, the teachers' own GT loss (the student's loss options), `precision_kl` and the frame tokens
    in the first-loss line"""
    n, mark = [0], [time.time(), 0]
    gb = lambda f: f() / 1e9 if torch.cuda.is_available() else 0.0  # noqa: E731
    first = {}                                                              # the teachers on the first batch of this process: their own GT loss, precision_kl, the frame tokens
    win = collections.deque(maxlen=LOG_WINDOW)                              # (gt, kd) of the last LOG_WINDOW calls: gt_mean / kd_mean of a log row = the means since the row before
    pwin = collections.deque(maxlen=LOG_WINDOW)                             # E395 addendum A1: the position loss that is trained (curve with --curve-w, else node) of the last LOG_WINDOW calls: pos_mean / pos_n of a log row

    def gt_loss(out, b, w=wire_w):
        return LJK.jtoken_loss(out, b, none_weight=none_weight, pos_loss=pos_loss, wire_w=w, curve_w=curve_w, node_w=node_w)

    def loss_fn(m, b):
        n[0] += 1
        dev = next(m.parameters()).device
        ns = None
        if noise_pos or noise_e:                                                # E381: the inputs of this step perturbed (`add_noise`, keyed on the head's own step counter)
            if n[0] == 1:                                                       # the CLEAN loss of the first batch of this process (eval: no step count; the head has no dropout): the twin check against the run without noise
                was = m.training
                m.eval()
                with torch.no_grad():
                    bc = to_device(b, dev)
                    first.update(clean=float(gt_loss(m(bc), bc)[0]), tokens=list(b.get("token") or []))
                torch.clear_autocast_cache()                                    # the no_grad forward filled autocast's cast-weight cache with copies WITHOUT grad_fn: the training forward below would reuse them (no grads)
                m.train(was)
            b, ns = add_noise(b, noise_pos, noise_e, noise_seed, m._step_py)
        b = to_device(b, dev)
        out = m(b)
        loss, info = gt_loss(out, b)
        if curve_w or node_w:                                                   # a single step's value is one batch of its own composition; the mean of the window is what a rule can read (pos_n = the calls in it: 100 unless this process started less than 100 calls ago)
            assert "curve" in info and "node" in info, "a position loss weight is set but the loss returned no curve / node info (the weights did not reach jtoken_loss)"
            pwin.append(float(info["curve" if curve_w else "node"]))
            info.update(pos_mean=float(np.mean(pwin)), pos_n=len(pwin))
        if ns is not None:
            info.update(noise_disp=ns[0], noise_e_abs=ns[1])
        if teachers:
            outs_t = teacher_forward(teachers, b)
            if n[0] == 1:
                assert all(torch.equal(o["cand"], out["cand"]) and torch.equal(o["valid"], out["valid"]) for o in outs_t), "a teacher saw other tokens than the student"
                first.update(gt=[float(gt_loss(o, b, 0.0)[0]) for o in outs_t], prec=precision_kl(outs_t, teacher_forward(teachers, b, amp=True)), tokens=list(b.get("token") or []))
            kd, kd_info = LJK.distill_loss(out, outs_t)
            info.update(kd_info, gt=float(loss))                                    # gt: the GT loss alone (the total, hist and final_loss hold W x kd too)
            win.append((info["gt"], info["kd"]))
            info.update(gt_mean=float(np.mean([g for g, _ in win])), kd_mean=float(np.mean([k for _, k in win])))
            loss = loss + distill_weight * kd
        info.update(gpu_gb=gb(torch.cuda.max_memory_allocated), gpu_res_gb=gb(torch.cuda.max_memory_reserved))
        if n[0] == 1 and run is not None:                                       # the loss of the start weights (before the first update)
            run.log(f"first loss of this process: {float(loss):.4f} (head step {m._step_py - 1} at this forward; before any update of this process)"
                    + (f"; GT loss {info['gt']:.4f} + KD {info['kd']:.4f} (x W {distill_weight}) = kd_s {info['kd_s']:.4f} + kd_l {info['kd_l']:.4f}, entropy of the teachers' mean per source short {info['ent_s']:.4f} / long {info['ent_l']:.4f}; "
                       f"the teachers' own GT loss on this batch {', '.join(f'{v:.4f}' for v in first['gt'])}; KL(the teachers' mean in float32 || under bf16 autocast) short {first['prec'][0]:.2e} / long "
                       f"{first['prec'][1]:.2e}; frame tokens of this batch {first['tokens']}" if teachers else "")
                    + (f"; input noise (E381) SP {noise_pos} SE {noise_e}: mean displacement of the valid tokens {ns[0]:.4f} m, mean |noise e| {ns[1]:.4f}; the CLEAN loss of this batch {first['clean']:.4f}; "
                       f"frame tokens of this batch {first['tokens']}" if ns is not None else ""))
        if n[0] == 10:
            mark[:] = [time.time(), 10]
        if n[0] == 50 or n[0] % 100 == 0:
            run.log(f"clock: {(time.time() - mark[0]) / (n[0] - mark[1]):.3f} s/step over the last {n[0] - mark[1]} steps of this process; over all {n[0]} steps so far: peak GPU allocated {info['gpu_gb']:.2f} GB, "
                    f"reserved {info['gpu_res_gb']:.2f} GB, valid tokens in the largest frame {float(b['valid'].sum(1).max()):.0f}; GT links outside the gate in this batch {info['gt_outside']:.0f} of "
                    f"{info['gt_links']:.0f}, ylong against the token chord class {info['gt_class']:.0f}; loss parts nll_s {info['nll_s']:.3f} nll_l {info['nll_l']:.3f} ce1 {info['ce1']:.3f} ce3 "
                    f"{info['ce3']:.3f} scale_l1 {info.get('scale_l1', float('nan')):.3f}"
                    + (f"; early reads (E392) nll1_s {info['nll1_s']:.3f} nll1_l {info['nll1_l']:.3f}" + (f" nll2_s {info['nll2_s']:.3f} nll2_l {info['nll2_l']:.3f}" if "nll2_s" in info else "") if wire_w else "")
                    + (f"; position head (E395) curve {info['curve']:.4f} node {info['node']:.4f} (link samples {info['n_curve']:.0f}, GT links {info['n_gl']:.0f}, with both nodes present {info['n_links']:.0f}, of which left out by the chord rule {info['n_excl']:.0f}), mean |dpos| {info['dpos_abs']:.3f} m, at the cap {info['dpos_cap']:.3f}" if curve_w or node_w else "")
                    + (f"; input noise: mean displacement {ns[0]:.4f} m, mean |noise e| {ns[1]:.4f}" if ns is not None else ""))
            mark[:] = [time.time(), n[0]]
        return loss, info
    return loss_fn


def load_model(run, dev, runs=None):
    """-> (the head of runs/<run>/model.pt in eval mode, its args)"""
    ck = torch.load(os.path.join(runs or T.RUNS, run, "model.pt"), map_location="cpu", weights_only=False)
    check_removed(ck["args"], f"run {run}")
    m = build_head(ck["args"])
    m.load_state_dict(ck["state"])
    return m.to(dev).eval(), ck["args"]


def load_teachers(args, dev):
    """E379 --distill-from: the teachers' heads (`load_model`: eval mode, their own step counter; every parameter frozen); refused when a teacher learned on other tokens than this run (cache seed, ident, K, cache root)
    or its step counter is not its run's steps (the key weights of another state). Their build draws random numbers: `train` seeds again right before it builds the student"""
    tok = lambda x: (x.get("seed"), x.get("ident"), x.get("K"), x.get("cache"))  # noqa: E731
    heads = []
    for p in distill_paths(args["distill_from"]):
        m, ta = load_model(os.path.basename(p), dev, os.path.dirname(p))
        assert tok(ta) == tok(args), f"--distill-from {p}: its tokens (cache seed, ident, K, cache root) {tok(ta)} are not this run's {tok(args)}"
        assert m._step_py == int(m.step) == ta["steps"], f"--distill-from {p}: its step counter {m._step_py} is not the {ta['steps']} steps of its run"
        heads.append(m.requires_grad_(False))
    return heads


def teacher_forward(teachers, b, amp=False):
    """the teachers' outputs on the batch b: eval, no_grad, float32 with autocast OFF (inside the trainer's autocast); amp=True: under bf16 autocast (the precision probe of the first batch)"""
    with torch.no_grad(), torch.autocast(b["valid"].device.type, dtype=torch.bfloat16, enabled=amp):
        return [t(b) for t in teachers]


def precision_kl(outs, outs_amp):
    """(short, long): KL(the teachers' mean in float32 || their mean under bf16 autocast), the precision effect of autocast on the targets (`jtoken.distill_loss` with the autocast mean as the student)"""
    i = LJK.distill_loss(dict(outs_amp[0], logp=LJK.teacher_logp([o["logp"] for o in outs_amp])), outs)[1]
    return i["kd_s"], i["kd_l"]


def first_batches(loader, batch_fn, seed, n=START_BATCHES):
    """the run's first n batches: the order `trainer.fit` draws after its seed_all(seed), from a COPY of the loader without workers (iterating the run's own loader would start its persistent workers, and fit's reset
    of that iterator draws another shuffle); it seeds the global RNG (`train` seeds again before the student is built, fit before the first batch)"""
    T.seed_all(seed)
    if batch_fn is not None:
        return [batch_fn(s) for s in range(n)]
    it = iter(torch.utils.data.DataLoader(loader.dataset, batch_size=loader.batch_size, shuffle=isinstance(loader.sampler, torch.utils.data.RandomSampler), num_workers=0, collate_fn=loader.collate_fn,
                                          drop_last=loader.drop_last))
    return [next(it) for _ in range(n)]


def teacher_start_loss(teachers, batches, dev, **loss_kw):
    """E379 hard start check: each teacher's own GT loss (`jtoken.jtoken_loss` with the student's loss options; `teacher_forward`: float32, eval) on each batch -> (teachers, batches) array"""
    return np.array([[float(LJK.jtoken_loss(o, b, **loss_kw)[0]) for o in teacher_forward(teachers, b)] for b in (to_device(x, dev) for x in batches)]).T


def train(a, runs=T.RUNS, ckpt_every=500, smoke_k=SMOKE_K):
    """the loop of the parsed arguments `a` (no GPU check: `main` has it); --seed chooses the B' cache (the loader's seed), the head seed (--head-seed, default --seed) the initialisation and the data order"""
    hs = a.seed if getattr(a, "head_seed", None) is None else a.head_seed
    T.seed_all(hs)
    args = dict(vars(a), steps=20 if a.smoke else a.steps,
                distill_sha256=[file_sha256(os.path.join(p, "model.pt")) for p in distill_paths(a.distill_from)] if a.distill_from else None)
    check_start(args, os.path.join(runs, a.name))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    teachers = load_teachers(args, dev) if a.distill_from else []
    geom = bool(a.curve_w or a.node_w)                                      # E395: the loss reads the GT geometry keys
    if a.smoke:
        loader, batch_fn = None, lambda step: JK.synthetic_batch(a.bs, smoke_k, seed=hs * 100003 + step, geom=geom)
    else:
        import lane_ptok_data as LP  # the lane chat's loader (card 2a); lazily, so that the tests and --smoke need no cache
        _, loader = LP.loader("train", a.seed, a.bs, a.workers, ident=a.ident, K=a.K, **({"cache": a.cache} if a.cache else {}), **({"geom": True} if geom else {}))
        batch_fn = None
    if teachers:
        tl = teacher_start_loss(teachers, first_batches(loader, batch_fn, hs), dev, none_weight=a.none_weight, pos_loss=a.pos_loss)
        assert all(TEACHER_GT[0] <= v <= TEACHER_GT[1] for v in tl.mean(1)), f"--distill-from: the teachers' own GT loss on the first {START_BATCHES} batches {tl.round(4).tolist()} has a mean outside {TEACHER_GT}"
    run = T.Run(a.name, root=runs)
    T.seed_all(hs)                                                          # the student's initialisation never depends on what drew before (the teachers' build, the start batches)
    model = build_head(args)
    run.log(f"jtoken bs {a.bs}: {args['steps']} steps, lr {a.lr}, trainable {JK.count_params(model)} parameters ({JK.count_params(model) / 1e6:.3f} M: the whole head, d {ARCH['d']} layers {ARCH['layers']} heads {ARCH['heads']} ffn {ARCH['ffn']}; B' is cached), "
            f"cache seed {a.seed}, head seed {hs}{' (--head-seed)' if hs != a.seed or getattr(a, 'head_seed', None) is not None else ''}, "
            + (f"none weight {a.none_weight}, " if a.none_weight != 1.0 else "")
            + (f"positive loss {a.pos_loss}, " if a.pos_loss != "sum" else "")
            + (f"message rounds (E392) {a.rounds}, messages {a.msg}, early wiring loss weight {a.wire_weight}, " if a.rounds else "")
            + (f"position head (E395) with curve loss weight {a.curve_w}, node loss weight {a.node_w}, " if a.pos_head else "")
            + (f"distill from {a.distill_from} (sha256 {', '.join(s[:12] for s in args['distill_sha256'])}, weight {a.distill_weight}), " if a.distill_from else "")
            + (f"input noise (E381) pos {a.noise_pos} m per axis, e {a.noise_e} (expected mean displacement {a.noise_pos * math.sqrt(math.pi / 2):.4f} m, mean |noise e| {a.noise_e * math.sqrt(2 / math.pi):.4f}), "
               if a.noise_pos or a.noise_e else "")
            + (f"synthetic batches, K {smoke_k}" if a.smoke else f"cache {a.cache or 'default root'}, ident {a.ident}, K {a.K or 'auto'}"))
    if teachers:
        run.log(f"teachers (float32, eval, step counters {[t._step_py for t in teachers]}): their own GT loss (the student's loss options) on the first {START_BATCHES} batches "
                + "; ".join(f"{p}: mean {row.mean():.4f} (one batch {row.min():.4f} - {row.max():.4f})" for p, row in zip(a.distill_from.split(","), tl)) + f"; the band of the mean {TEACHER_GT}: ok")
    T.fit(model, make_loss(run, a.none_weight, a.pos_loss, teachers, a.distill_weight, a.noise_pos, a.noise_e, hs, a.wire_weight, a.curve_w, a.node_w), run, steps=args["steps"], loader=loader, batch_fn=batch_fn, lr=a.lr, sched="cosine", warmup=0.05,
          ckpt_every=ckpt_every, seed=hs, args=args, device=dev, config=dict(design="jtoken"))
    run.log(f"done {a.name}")


def main(argv=None, runs=T.RUNS):
    a = parse_args(argv)
    assert a.smoke or torch.cuda.is_available(), "no GPU visible: a run of the card trains on the GPU (bf16); only --smoke (20 steps, a plumbing test) may run on the CPU"
    train(a, runs)


if __name__ == "__main__":
    main()
