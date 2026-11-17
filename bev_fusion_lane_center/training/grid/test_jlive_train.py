"""Tests of the S4 joint trainer (`train/jlive_train.py`, E372) on the CPU: a toy `QGraph` as B' (a run folder with its model.pt, as `lane_live_tok`'s selftest builds it), a small `JTokenHead` past the key-weight switch as the champion, frames whose GT tables are
made of the toy's own tokens, K 64; no cache, no GPU.

    wsl bash tools/env/_py14.sh test_jlive_train.py            # the selftest, the real-data check (the E365 cache; skipped with the reason when it is missing), then the hand mutants of jlive_train.py ONE AT A TIME (a shared box): each must make the selftest fail
    wsl bash tools/env/_py14.sh test_jlive_train.py selftests  # the selftest and the real-data check only

The rules by hand (the kill rules on records, AdamW's step against torch's, the shares on a known case); one step by hand in each arm (the loss as defined); 2-step runs through `train` in each arm (what moves and what does not, the per-group clip between two spy hooks,
the logged numbers, the frames, model.pt read back, the export check); every kill; a step skipped by LabelOverflow = the run without it; the resume bit for bit and flagged, the roll-back flagged, every recipe key refused at a resume.
A mutant = one textual change of jlive_train.py; the mutated copy is loaded in place of the file (`one <path>`); KILLED = the child failed with an AssertionError or a `Kill` (its last line is printed).
"""
import copy
import json
import math
import os
import subprocess
import sys
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""                                                      # the CPU only: `train` takes the GPU when it sees one (before the first CUDA call)
from torch.optim.optimizer import register_optimizer_step_pre_hook  # noqa: E402

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub_ in ("training", "training/grid", "training/data", "training/data/prep", "training/train", "evaluation"):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
TRAIN = os.path.join(ROOT, "training", "train", "jlive_train.py")
ENV = dict(os.environ, BFLC_ROOT=ROOT, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="")      # a shared box: 2 threads, one child at a time, no GPU
SOFT = ["mnms", 1.75, 0.1]
KT = 64                                                                                      # tokens per frame of the toy (the trainer's K = 512 is checked, then patched)

MUTANTS = [
    ("the head's live_inputs is not checked", "        if m.head.live_inputs != live:\n", "        if False:\n"),
    ("a zero gradient of the head's loss on the node head is accepted in s4", "            if live and not n_h > 0:\n", "            if False:\n"),
    ("arm b accepts a head whose loss reaches the node head", "            if not live and n_h != 0:\n", "            if False:\n"),
    ("B' is not a submodule (not clipped, not checkpointed)", "        self.head, self.src = head, src\n", "        self.head = head\n        object.__setattr__(self, \"src\", src)\n"),
    ("one joint clip over both groups", "nh, nn_ = (torch.nn.utils.clip_grad_norm_([p for p in ps if p.grad is not None], CLIP) for ps in (self.heads, self.nodes))",
     "nh = nn_ = torch.nn.utils.clip_grad_norm_([p for p in self.heads + self.nodes if p.grad is not None], CLIP)"),
    ("the node head is not clipped", "for ps in (self.heads, self.nodes))", "for ps in (self.heads, []))"),
    ("fit's joint clip is on", "clip=0, ckpt_every=ckpt_every", "clip=CLIP, ckpt_every=ckpt_every"),
    ("the frozen check is skipped", "            if held:\n", "            if False:\n"),
    ("the learning check is skipped", "            if not (math.isfinite(u) and u > 0):\n", "            if False:\n"),
    ("a LabelOverflow step is not counted", "            m.rec[\"overflow\"].append(i)\n", "            pass\n"),
    ("a LabelOverflow step is trained", "            return torch.full((), float(\"nan\")), dict(", "            return 0.0 * sum(p.sum() for p in m.head.parameters()), dict("),
    ("the overflow kill is off", "    if done >= OVERFLOW_FROM and n > OVERFLOW_MAX * done:\n", "    if False:\n"),
    ("the overflow kill acts before step 100", "    if done >= OVERFLOW_FROM and n > OVERFLOW_MAX * done:\n", "    if n > OVERFLOW_MAX * done:\n"),
    ("the B-rise kill reads the whole curve", "_mean(b[-WIN:]) > B_RISE * _mean(b[:WIN])", "_mean(b) > B_RISE * _mean(b[:WIN])"),
    ("the B-rise kill acts in z", "    if train_nodes and done > WIN and", "    if done > WIN and"),
    ("the head-rise kill reads the last 200 steps", "_mean(h[lo - 1:hi]) > H_RISE * _mean(h[:WIN])", "_mean(h[-WIN:]) > H_RISE * _mean(h[:WIN])"),
    ("the head-rise kill is off", "    if done >= hi and", "    if False and"),
    ("B's loss is not in the loss of s4", "        loss = loss_h + w * loss_b if tn else loss_h\n", "        loss = loss_h\n"),
    ("B's loss is in the loss of z", "        loss = loss_h + w * loss_b if tn else loss_h\n", "        loss = loss_h + w * loss_b\n"),
    ("B' trains in z", "LLT.LiveSource(args[\"b_run\"], K, train_nodes=bool(args[\"train_nodes\"]))", "LLT.LiveSource(args[\"b_run\"], K, train_nodes=True)"),
    ("the head run's none weight is ignored", "none_weight=g(\"none_weight\") if a.none_weight is None", "none_weight=1.0 if a.none_weight is None"),
    ("the data order is B''s seed", "seed=args[\"head_seed\"], args=args,", "seed=a.seed, args=args,"),
    ("no share at the last step", "if tn and live and (i % SHARE_EVERY == 0 or i == steps):", "if tn and live and i % SHARE_EVERY == 0:"),
    ("the share takes the clipped gradient", "        if cur.get(\"gb\") is not None:\n            sh = shares(", "        nh, nn_ = (torch.nn.utils.clip_grad_norm_([p for p in ps if p.grad is not None], CLIP) for ps in (self.heads, self.nodes))\n        if cur.get(\"gb\") is not None:\n            sh = shares("),
    ("the share counts B''s gradient as the head's", "        hd = p.grad - b\n", "        hd = p.grad\n"),
    ("a non-finite loss is not a kill", "        if not (math.isfinite(float(loss_h)) and math.isfinite(float(loss_b))):\n", "        if False:\n"),
    ("the frames are not logged", "        if i <= FRAMES_LOGGED:\n", "        if False:\n"),
    ("a resume is not flagged", "        if st[\"last\"] is None and i > 1:\n", "        if False:\n"),
    ("a roll-back is not flagged", "        elif st[\"last\"] is not None and i <= st[\"last\"]:\n", "        elif False:\n"),
    ("the record is not checkpointed", "    def get_extra_state(self):\n        return {k: list(v) for k, v in self.rec.items()}", "    def get_extra_state(self):\n        return dict(b=[], h=[], overflow=[])"),
    ("model.pt holds the start B'", "b_state=joint.src.model.state_dict(), b_args=b_args", "b_state=start, b_args=b_args"),
    ("the export does not check the frozen tensors", "        assert not moved, f\"model.pt: B''s", "        assert True, f\"model.pt: B''s"),
    ("an export failure keeps .done", "            os.replace(run.path(\".done\"), run.path(\".done.FALSE\"))", "            pass"),
    ("AdamW's step without the bias correction of m", "return group[\"lr\"] / (1 - b1 ** t) * m", "return group[\"lr\"] * m"),
    ("K is 384", "K = 512 ", "K = 384 "),
    ("the recipe lacks b_weight", "\"live_inputs\", \"b_weight\", \"none_weight\"", "\"live_inputs\", \"none_weight\""),
]


def load(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def kills(fn, text):
    """fn must stop with `Kill` (a RuntimeError) whose message holds `text`"""
    try:
        fn()
    except RuntimeError as e:
        assert type(e).__name__ == "Kill" and text in str(e), (text, repr(e))
        return
    raise AssertionError(f"not killed although it must be ({text})")


def refuses(fn, text):
    try:
        fn()
    except AssertionError as e:
        assert text in str(e), (text, str(e))
        return
    raise AssertionError(f"accepted although it must be refused ({text})")


class NoRun:
    """the stand-in of `trainer.Run` for the direct calls: the log lines and the status"""

    def __init__(self):
        self.lines, self._status = [], {}

    def log(self, *a):
        self.lines.append(" ".join(str(x) for x in a))

    def status(self, **kw):
        self._status.update(kw)


# ------------------------------------------------------------------------------------------------------------------------------ the toy world
def world(tmp):
    """a toy B' run (tmp/b), a head run tmp/h (seed 0, none weight 0.25, head seed 2, past the switch), batches A (2 frames, GT tables without links) and B (2 frames, a linked GT pair on the toy's own tokens) -> dict"""
    import numpy as np
    import torch
    import jtok as JT
    import jtoken as JK
    import lane_live_tok as LLT
    import qgraph as QG
    torch.manual_seed(0)
    m = QG.QGraph(width=16, soft_nms=SOFT)
    with torch.no_grad():
        m.nodes.head1.bias[0] = 3.0                                                         # many valid tokens (as the selftest of lane_live_tok)
    os.makedirs(os.path.join(tmp, "b"))
    torch.save(dict(state=m.state_dict(), args=dict(width=16, soft_nms=SOFT)), os.path.join(tmp, "b", "model.pt"))
    head = JK.JTokenHead(d=32, heads=4, layers=2, ffn=64)
    head.set_step(12000)
    os.makedirs(os.path.join(tmp, "h"))
    torch.save(dict(state=head.state_dict(), args=dict(seed=0, ident="e292_a", d=32, layers=2, heads=4, ffn=64, none_weight=0.25, head_seed=2, teacher="none"), steps=12000), os.path.join(tmp, "h", "model.pt"))
    g = torch.Generator().manual_seed(1)
    xs = [torch.randn(39, 200, 200, generator=g).half() for _ in range(4)]
    with torch.no_grad():
        tk = LLT.LiveSource(m, KT, train_nodes=False).eval().tokens(torch.stack(xs))[1]
    allowed = JT.pair_geometry(tk["pos"], tk["dirv"], tk["valid"])[1]
    items = []
    for f in range(4):
        if f < 2:
            t = LLT.toy_table(np.zeros((0, 2)), [], [])
        else:
            i, j = [int(v) for v in allowed[f].nonzero()[0]]
            t = LLT.toy_table(tk["pos"][f][[i, j]].double().numpy(), [(0, 1)], [False])
            t["dirv"] = tk["dirv"][f][[i, j]].double().numpy()
        items.append(dict(x=xs[f], token=f"f{f}", row=f, table=t))
    return dict(b_dir=os.path.join(tmp, "b"), A=LLT.collate(items[:2]), B=LLT.collate(items[2:]))


def cli(w, name, train_nodes, live=1, steps=2, extra=()):
    return ["--name", name, "--seed", "0", "--head-run", "h", "--b-run", w["b_dir"], "--train-nodes", str(train_nodes), "--live-inputs", str(live), "--steps", str(steps), "--bs", "2", "--workers", "0"] + list(extra)


class Built:
    """inside the block `T.batches` hands out `batches` and `T.build` records the Joint it builds"""

    def __init__(self, T, batches):
        self.T, self.batches, self.joints = T, batches, []

    def __enter__(self):
        self.old = self.T.batches, self.T.build
        self.T.batches = lambda *a, **k: self.batches

        def build(args, runs):
            j, ba = self.old[1](args, runs)
            self.joints.append(j)
            return j, ba
        self.T.build = build
        return self

    def __exit__(self, *a):
        self.T.batches, self.T.build = self.old


def direct(T, w, tmp, train_nodes, live=1):
    """the Joint, the loss of one step and the hook of an arm, built as `train` builds them (no fit)"""
    args = T.resolve(T.parse_args(cli(w, "d", train_nodes, live)), tmp)
    joint = T.build(args, tmp)[0]
    run = NoRun()
    hook = T.StepHook(run, joint)
    joint.train()
    return joint, T.make_loss(run, joint, args, hook), hook, run, args


# ------------------------------------------------------------------------------------------------------------------------------ the rules by hand
def rules(T):
    import numpy as np
    import torch
    nan = float("nan")
    rec = lambda b, h=None, ov=(): dict(b=list(b), h=list(h if h is not None else [1.0] * len(b)), overflow=list(ov))  # noqa: E731
    says = lambda v, text: v is not None and text in v  # noqa: E731
    assert T.verdict(rec([1.0] * 99, ov=range(5)), True) is None, "overflow: judged from step 100 on"
    assert T.verdict(rec([1.0] * 100, ov=[3]), True) is None and says(T.verdict(rec([1.0] * 100, ov=[3, 4]), True), "LabelOverflow"), "overflow: 2 of 100 > 1 %"
    assert T.verdict(rec([1.0] * 200, ov=[1, 2]), False) is None and says(T.verdict(rec([1.0] * 200, ov=[1, 2, 3]), False), "LabelOverflow")
    assert T.verdict(rec([1.0] * 200 + [1.19] * 200), True) is None and says(T.verdict(rec([1.0] * 200 + [1.21] * 200), True), "B''s node loss rose"), "B rise: above 1.2 x the start mean"
    assert says(T.verdict(rec([1.0] * 200 + [2.0] * 50), True), "B''s node loss rose"), "B rise: any window of the most recent 200 steps after step 200 (here steps 51 - 250)"
    assert T.verdict(rec([1.0] * 200 + [2.0] * 50), False) is None, "B rise: not in z (B' frozen)"
    assert says(T.verdict(rec([1.0] * 199 + [nan] + [1.25] * 199 + [nan]), True), "B''s node loss rose"), "B rise: a skipped step (NaN) is left out of the means"
    late = lambda tail: [1.0] * 1499 + tail  # noqa: E731
    assert T.verdict(rec([1.0] * 2000, late([1.09] * 501)), True) is None and says(T.verdict(rec([1.0] * 2000, late([1.11] * 501)), True), "the head's loss"), "head rise: steps 1500 - 2000 above 1.10 x steps 1 - 200"
    assert T.verdict(rec([1.0] * 2000, late([1.0] * 301 + [1.25] * 200)), True) is None, "head rise: the window is steps 1500 - 2000, not the last 200"
    assert T.verdict(rec([1.0] * 1999, late([1.5] * 500)), True) is None, "head rise: judged once step 2000 is done"
    # AdamW's step, by hand, against torch's (weight decay 0): the first step (no state) and the second
    p = torch.tensor([0.3, -1.2, 2.0], requires_grad=True)
    opt = torch.optim.AdamW([p], lr=0.01, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.0)
    for g in (torch.tensor([1.0, -2.0, 0.5]), torch.tensor([0.2, 0.3, -4.0])):
        p.grad = g
        u = T.adam_update(g, opt.state.get(p, {}), opt.param_groups[0])
        before = p.detach().clone()
        opt.step()
        assert torch.allclose(before - p.detach(), u, atol=1e-7, rtol=1e-5), (before - p.detach(), u)
    # the shares on a known case: after one step with gradient (1, 2) v_hat = (1, 4); full gradient (3, 4), B's part (1, 0): head part (2, 4) -> preconditioned (2, 2), B (1, 0): ushare = 2.828 / 3.828; gshare = 4.472 / 5.472
    q = torch.zeros(2, requires_grad=True)
    oq = torch.optim.AdamW([q], lr=0.1, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.0)
    q.grad = torch.tensor([1.0, 2.0])
    nostate = T.shares([q], [torch.tensor([1.0, 0.0])], oq)
    assert math.isnan(nostate["ushare"]) and abs(nostate["gshare"] - 2 / 3) < 1e-6, nostate                                    # head part (0, 2), B's (1, 0), no state: no ushare
    oq.step()
    q.grad = torch.tensor([3.0, 4.0])
    s = T.shares([q], [torch.tensor([1.0, 0.0])], oq)
    assert abs(s["ushare"] - math.sqrt(8) / (math.sqrt(8) + 1)) < 1e-5 and abs(s["gshare"] - math.sqrt(20) / (math.sqrt(20) + 1)) < 1e-6, s
    assert T.K == 512 and T.CLIP == 3.0 and T.WIN == 200 and T.B_RISE == 1.2 and T.H_LATE == (1500, 2000) and T.H_RISE == 1.10 and T.SHARE_EVERY == 200 and T.LOG_EVERY == 100 and np.isclose(T.OVERFLOW_MAX, 0.01)


# ------------------------------------------------------------------------------------------------------------------------------ one step by hand, the kills, the flags
def steps_by_hand(T, w, tmp):
    import numpy as np
    import torch
    import jtoken as JK
    import lane_live_tok as LLT
    B = w["B"]
    for tn, live in ((1, 1), (0, 1), (1, 0)):
        joint, lf, hook, run, args = direct(T, w, tmp, tn, live)
        ref = copy.deepcopy(joint)
        loss, info = lf(joint, B)
        no, tk = ref.src.tokens(B["x"])
        lh, _ = JK.jtoken_loss(ref.head(LLT.head_inputs(tk)), ref.src.labelled(tk, B["table"], B["token"], B["row"], gate=False), none_weight=0.25, pos_loss="sum")
        lb, _ = LLT.b_loss(no, B["targets"])
        want = lh + lb if tn else lh
        assert abs(float(loss) - float(want)) < 1e-6 and abs(info["head"] - float(lh)) < 1e-6 and abs(info["b_loss"] - float(lb)) < 1e-6, f"arm {T.arm(tn, live)}: the loss of a step is not the head's loss {'+ B''s' if tn else 'alone'} ({float(loss)} {float(want)})"
        assert loss.requires_grad and (not tn or not live or any("> 0: s4" in x for x in run.lines)) and (live or any("0: arm b" in x for x in run.lines)), (T.arm(tn, live), run.lines)
        assert hook.cur["i"] == 1 and hook.cur["info"] is info and hook.cur["gb"] is None and not hook.cur["logged"] and any("step 1 frames ['f2', 'f3']" in x for x in run.lines)
        lab = ref.src.labelled(tk, B["table"], B["token"], B["row"], gate=False)["lab"]
        assert info["n_valid"] == float(np.mean([x["n_valid"] for x in lab])) and info["gt_cov"] == sum(x["n_gt_assigned"] for x in lab) / max(sum(x["n_gt"] for x in lab), 1) and info["pair_cov"] == info["pos_pairs"] / max(info["gt_links"], 1.0) and info["overflow"] == 0.0
        assert len(joint.rec["b"]) == 1 and abs(joint.rec["h"][0] - float(lh)) < 1e-6
    # the live check every step: s4 with a head that cuts its inputs, b with a live head
    joint, lf, *_ = direct(T, w, tmp, 1, 1)
    joint.head.live_inputs = False
    kills(lambda: lf(joint, B), "live_inputs")
    joint, lf, *_ = direct(T, w, tmp, 1, 0)
    joint.head.live_inputs = True
    kills(lambda: lf(joint, B), "live_inputs")
    # the start check: s4 whose head's loss does not reach B (the tokens cut before the head), b whose head's loss does (the head's cut switched off)
    old = LLT.head_inputs
    LLT.head_inputs = lambda tk: {k: v.detach() for k, v in old(tk).items()}
    try:
        joint, lf, *_ = direct(T, w, tmp, 1, 1)
        kills(lambda: lf(joint, B), "refused at the start")
    finally:
        LLT.head_inputs = old
    oc = JK._const
    JK._const = lambda t: t
    try:
        joint, lf, *_ = direct(T, w, tmp, 1, 0)
        kills(lambda: lf(joint, B), "arm b: the head's loss reaches")
    finally:
        JK._const = oc
    # the hook: a frozen tensor of B' with a gradient at the first update (z: the node head), a zero head update on a logged step
    joint, lf, hook, run, _ = direct(T, w, tmp, 0, 1)
    opt = torch.optim.AdamW(list(joint.head.parameters()), lr=1e-4)
    for p in joint.head.parameters():
        p.grad = torch.ones_like(p)
    nd = next(joint.src.model.nodes.parameters())
    nd.grad = torch.ones_like(nd)
    kills(lambda: hook(opt, (), {}), "hold a gradient")
    nd.grad = None
    hook(opt, (), {})                                                                       # nothing held: passes, the check is done once
    for p in joint.head.parameters():
        p.grad = torch.zeros_like(p)
    hook.cur = dict(i=100, info={}, gb=None, logged=True)
    kills(lambda: hook(opt, (), {}), "not learning")
    # LabelOverflow: the step returns a NaN loss (fit's skip: no optimiser step) and is counted; the kill above 1 % from step 100 on
    pmax, LLT.LC.PMAX = LLT.LC.PMAX, 0
    try:
        joint, lf, _, run, _ = direct(T, w, tmp, 1, 1)
        loss, info = lf(joint, B)
        assert math.isnan(float(loss)) and joint.rec["overflow"] == [1] and math.isnan(joint.rec["b"][0]) and info["overflow"] == 1.0 and any("LabelOverflow" in x for x in run.lines)
        joint.rec = dict(b=[1.0] * 200, h=[1.0] * 200, overflow=[7])
        assert math.isnan(float(lf(joint, B)[0])) and joint.rec["overflow"] == [7, 201], "2 of 201 steps: no kill"
        joint.rec = dict(b=[1.0] * 100, h=[1.0] * 100, overflow=[7])
        kills(lambda: lf(joint, B), "LabelOverflow")
    finally:
        LLT.LC.PMAX = pmax
    # a loss that is not finite stops the run (fit alone would skip the step): B''s loss NaN, in z too
    ob = LLT.b_loss
    LLT.b_loss = lambda no, tg: (torch.tensor(float("nan")), {})
    try:
        for tn in (1, 0):
            joint, lf, *_ = direct(T, w, tmp, tn, 1)
            kills(lambda: lf(joint, B), "not finite")
    finally:
        LLT.b_loss = ob
    # the B-rise kill through the step (s4: B's loss of the toy is far above 1.2 x a start mean of 1), not in z
    joint, lf, *_ = direct(T, w, tmp, 1, 1)
    joint.rec = dict(b=[1.0] * 399, h=[1.0] * 399, overflow=[])
    kills(lambda: lf(joint, B), "B''s node loss rose")
    joint, lf, *_ = direct(T, w, tmp, 0, 1)
    joint.rec = dict(b=[1.0] * 399, h=[1.0] * 399, overflow=[])
    lf(joint, B)
    # the flags: the first step of a process after step 5 is a resume, a step number that does not grow a roll-back (both in status)
    joint, lf, _, run, _ = direct(T, w, tmp, 0, 1)
    joint.rec = dict(b=[1.0] * 5, h=[1.0] * 5, overflow=[])
    lf(joint, B)
    joint.rec = dict(b=[1.0] * 5, h=[1.0] * 5, overflow=[])
    lf(joint, B)
    assert run._status == dict(resumed_at=[5], rollbacks_at=[5]) and any("RESUMED at step 5" in x for x in run.lines) and any("ROLL-BACK to step 5" in x for x in run.lines), (run._status, run.lines)


# ------------------------------------------------------------------------------------------------------------------------------ runs through train
def runs(T, w, tmp):
    import torch
    import jtoken_train as JKT
    import trainer
    import lane_live_tok as LLT
    seen, real_fit = {}, trainer.fit
    rd = lambda n: [json.loads(x) for x in open(os.path.join(tmp, n, "log.jsonl"))]  # noqa: E731
    text = lambda n: open(os.path.join(tmp, n, "train.log"), encoding="utf-8").read()  # noqa: E731
    start = torch.load(os.path.join(w["b_dir"], "model.pt"))["state"]
    h0 = torch.load(os.path.join(tmp, "h", "model.pt"))["state"]
    # s4: two steps on batch B; two global pre-step hooks around the trainer's: `before` sees the gradients of fit's backward (fit's clip off), `after` those after the trainer's per-group clip
    seen.update(before=[], after=[])
    with Built(T, [w["B"]]) as bt:
        state = {}

        def before(opt, a, k):
            if "n" in state:
                seen["before"].append(state["n"]())

        hb = register_optimizer_step_pre_hook(before)

        def fit(model, loss_fn, run, **kw):
            groups = dict(head=list(model.head.parameters()), nodes=model.src.trainable())
            state["n"] = lambda: {g: math.sqrt(sum(float(p.grad.double().square().sum()) for p in ps if p.grad is not None)) for g, ps in groups.items()}
            ha = register_optimizer_step_pre_hook(lambda opt, a, k: seen["after"].append(state["n"]()))
            try:
                seen["fit_kw"] = kw
                return real_fit(model, loss_fn, run, **kw)
            finally:
                ha.remove()
        T.T.fit = fit
        try:
            T.train(T.parse_args(cli(w, "s4", 1)), runs=tmp, ckpt_every=500)
        finally:
            T.T.fit = real_fit
            hb.remove()
        j4 = bt.joints[0]
    assert len(seen["before"]) == len(seen["after"]) == 2, seen
    for bf, af in zip(seen["before"], seen["after"]):
        assert bf["nodes"] > T.CLIP and math.hypot(bf["head"], bf["nodes"]) > T.CLIP, f"the test needs the clip active on the node head and fit's own clip off: {bf}"
        for g in ("head", "nodes"):
            assert abs(af[g] - min(bf[g], T.CLIP * bf[g] / (bf[g] + 1e-6))) < 1e-4 * max(bf[g], 1e-3), f"the {g} group is not clipped on its own: before {bf} after {af}"
    rows = rd("s4")
    last = rows[-1]
    assert last["step"] == 2 and abs(last["clip_nodes"] - min(1.0, T.CLIP / (seen["before"][-1]["nodes"] + 1e-6))) < 1e-5 and abs(last["clip_head"] - min(1.0, T.CLIP / (seen["before"][-1]["head"] + 1e-6))) < 1e-5, last
    assert abs(last["grad_norm"] - math.hypot(seen["before"][-1]["head"], seen["before"][-1]["nodes"])) < 1e-3 * last["grad_norm"], "fit measures the joint norm (its clip off)"
    for k in ("head", "b_loss", "b_e", "b_mean", "label_s", "overflow", "pair_cov", "n_valid", "gt_cov", "upd_head", "nll_s", "ce1"):
        assert k in last and math.isfinite(last[k]), (k, last)
    assert "ushare" in last and "gshare" in last, "s4: the shares at the last step"
    assert 0 < last["ushare"] < 1 and 0 < last["gshare"] < 1 and last["upd_head"] > 0 and last["overflow"] == 0, last
    cfg = json.load(open(os.path.join(tmp, "s4", "config.json")))
    assert cfg["design"] == "jlive" and cfg["seed"] == 2 and cfg["clip"] == 0 and cfg["clip_per_group"] == 3.0 and cfg["args"]["none_weight"] == 0.25 and cfg["args"]["pos_loss"] == "sum" and cfg["args"]["head_seed"] == 2, cfg
    assert [id(p) for p in seen["fit_kw"]["params"]] == [id(p) for p in list(j4.head.parameters()) + j4.src.trainable()] and seen["fit_kw"]["sched"] == "cosine" and seen["fit_kw"]["lr"] == 1e-4, "fit optimises the head and the node head, nothing frozen"
    tx = text("s4")
    assert "jlive arm s4" in tx and "live_inputs True" in tx and "head h" in tx and w["b_dir"] in tx and "step 1 frames ['f2', 'f3']" in tx and "step 2 frames ['f2', 'f3']" in tx and "> 0: s4" in tx and "step 2: update share" in tx, tx[-1500:]
    # model.pt: the live head in jtoken_train's format, B' as trained; what moved per arm
    ck = torch.load(os.path.join(tmp, "s4", "model.pt"), weights_only=False)
    assert ck["args"]["live_inputs"] == 1 and ck["args"]["train_nodes"] == 1 and ck["args"]["head_run"] == "h" and ck["args"]["b_run"] == w["b_dir"] and ck["b_args"]["soft_nms"] == SOFT and ck["rec"]["overflow"] == [] and len(ck["rec"]["b"]) == 2
    hm, _ = JKT.load_model("s4", "cpu", tmp)
    assert hm.live_inputs and all(torch.equal(hm.state_dict()[k], j4.head.state_dict()[k].cpu()) for k in hm.state_dict())

    def moved(run):
        b = T.load_b(run, "cpu", tmp).state_dict()
        h = JKT.load_model(run, "cpu", tmp)[0].state_dict()
        return sorted({k.split(".")[0] for k in b if not torch.equal(b[k], start[k])}), any(not torch.equal(h[k], h0[k]) for k in h if k != "step")
    assert moved("s4") == (["nodes"], True), moved("s4")
    with Built(T, [w["B"]]) as bt:
        T.train(T.parse_args(cli(w, "z", 0)), runs=tmp, ckpt_every=500)
        T.train(T.parse_args(cli(w, "bb", 1, live=0)), runs=tmp, ckpt_every=500)
    assert moved("z") == ([], True) and moved("bb") == (["nodes"], True), (moved("z"), moved("bb"))
    assert "ushare" not in rd("z")[-1] and "ushare" not in rd("bb")[-1] and rd("z")[-1]["b_loss"] > 0 and "jlive arm z" in text("z") and "jlive arm b" in text("bb") and "0: arm b" in text("bb"), "z and b: no share; B''s loss logged"
    assert not JKT.load_model("bb", "cpu", tmp)[0].live_inputs
    # the export check: a frozen tensor of B' that moved (here: a start that differs) is refused and fit's .done becomes .done.FALSE
    jz = bt.joints[0]
    st = {k: v.clone() for k, v in start.items()}
    st[next(k for k in st if k.startswith("pyr.") and st[k].is_floating_point())] += 1.0
    run_z = trainer.Run("z", root=tmp)
    refuses(lambda: T.export(run_z, jz, json.load(open(os.path.join(tmp, "z", "config.json")))["args"], {"width": 16, "soft_nms": SOFT}, st), "must stay as at the start")
    assert os.path.exists(os.path.join(tmp, "z", ".done.FALSE")) and not os.path.exists(os.path.join(tmp, "z", ".done"))
    # a step skipped by LabelOverflow is the run without it: [A, B] with B over PMAX = the one-step run on [A] (all but the head's forward counter)
    pmax, LLT.LC.PMAX = LLT.LC.PMAX, 0
    try:
        with Built(T, [w["A"], w["B"]]):
            T.train(T.parse_args(cli(w, "ov", 1)), runs=tmp, ckpt_every=500)
        with Built(T, [w["A"]]):
            T.train(T.parse_args(cli(w, "one", 1, steps=1)), runs=tmp, ckpt_every=500)
    finally:
        LLT.LC.PMAX = pmax
    ov, one = (torch.load(os.path.join(tmp, n, "model.pt"), weights_only=False) for n in ("ov", "one"))
    assert ov["rec"]["overflow"] == [2] and math.isnan(ov["rec"]["b"][1]) and json.load(open(os.path.join(tmp, "ov", "status.json")))["updates"] == 1
    assert all(torch.equal(ov["b_state"][k], one["b_state"][k]) for k in one["b_state"]) and all(torch.equal(ov["state"][k], one["state"][k]) for k in one["state"] if k != "step"), "the skipped step changed a weight"
    # the resume: killed at step 3, resumed from the checkpoint of step 2 = the straight 4-step run, bit for bit; flagged in the log and in status.json
    with Built(T, [w["B"]]):
        T.train(T.parse_args(cli(w, "st", 1, steps=4)), runs=tmp, ckpt_every=2)
        real_make, n = T.make_loss, [0]

        def flaky(run, joint, args, hook):
            f = real_make(run, joint, args, hook)

            def g(m, b):
                n[0] += 1
                if n[0] == 3:
                    raise RuntimeError("killed")
                return f(m, b)
            return g
        T.make_loss = flaky
        try:
            try:
                T.train(T.parse_args(cli(w, "rs", 1, steps=4)), runs=tmp, ckpt_every=2)
                raise AssertionError("the run was not killed")
            except RuntimeError as e:
                assert "killed" in str(e), e
            T.make_loss = real_make
            T.train(T.parse_args(cli(w, "rs", 1, steps=4)), runs=tmp, ckpt_every=2)
        finally:
            T.make_loss = real_make
    a_, b_ = (torch.load(os.path.join(tmp, n_, "model.pt"), weights_only=False) for n_ in ("st", "rs"))
    for k in ("state", "b_state"):
        assert all(torch.equal(a_[k][x], b_[k][x]) for x in a_[k]), f"the resumed run is not the straight run ({k})"
    assert a_["rec"] == b_["rec"] and len(a_["rec"]["b"]) == 4, (a_["rec"], b_["rec"])
    assert json.load(open(os.path.join(tmp, "rs", "status.json")))["resumed_at"] == [2] and "RESUMED at step 2" in text("rs") and "resumed_at" not in json.load(open(os.path.join(tmp, "st", "status.json")))
    # a resume with another recipe is refused (every recipe key)
    cfg, folder = json.load(open(os.path.join(tmp, "st", "config.json")))["args"], os.path.join(tmp, "st")
    T.check_start(cfg, folder)
    ch = dict(seed=lambda v: v + 1, head_run=lambda v: v + "x", head_sha256=lambda v: "0" * 64, b_run=lambda v: v + "x", b_sha256=lambda v: "0" * 64, train_nodes=lambda v: 1 - v, live_inputs=lambda v: 1 - v, b_weight=lambda v: v * 2,
              none_weight=lambda v: v * 2, none_weight_near=lambda v: 0.5, pos_loss=lambda v: "max", head_seed=lambda v: v + 1, bs=lambda v: v + 1, steps=lambda v: v + 1, lr=lambda v: v * 2)
    assert set(ch) == set(T.RECIPE), f"a recipe key without a refusal test: {set(ch) ^ set(T.RECIPE)}"
    for k in ch:
        refuses(lambda: T.check_start(dict(cfg, **{k: ch[k](cfg[k])}), folder), "of another run")


def selftest(train_path=TRAIN):
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    assert not torch.cuda.is_available(), "the tests run on the CPU (CUDA_VISIBLE_DEVICES is set empty above)"
    T = load(train_path, "jlive_train_under_test")
    rules(T)
    T.K = KT                                                                                # the toy's tokens per frame (the real K is checked in `rules`)
    with tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "temp")) as tmp:
        w = world(tmp)
        steps_by_hand(T, w, tmp)
        runs(T, w, tmp)
    print("jlive_train selftest ok", flush=True)


def real_check(train_path=TRAIN):
    """the real data (needs the E365 cache and the lane GT store): `batches` checks that the B' run is the checkpoint of the cache (seed 0's B' accepted, seed 1's refused) and hands out the live loader's batches (x, table, token, row, targets)"""
    import lane_ptok_data as LP
    T = load(train_path, "jlive_train_real")
    if not os.path.exists(os.path.join(LP.cache_dir("train", 0, "e292_a"), "meta.json")):
        print(f"real check SKIPPED: no E365 TRAIN cache of B' seed 0 at {LP.cache_dir('train', 0, 'e292_a')}")
        return
    refuses(lambda: T.batches(0, 2, 0, "runs/e292_a_s1", "e292_a"), "is not the checkpoint that the cache")
    b = next(iter(T.batches(0, 2, 0, "runs/e292_a_s0", "e292_a")))
    assert sorted(b) == ["row", "table", "targets", "token", "x"] and b["x"].shape == (2, 39, 200, 200) and len(b["table"]) == 2, sorted(b)
    print("real check ok: the B' run is the cache's checkpoint, the live loader's batches")


def run_mutants():
    import test_lane_live as TLL                                                            # its launcher: numpy / torch imported with the project paths behind site-packages (the 319 paths cost 50 s)
    left = []
    with tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "temp")) as tmp:
        src = open(TRAIN, encoding="utf-8").read()
        for k, (name, a, b) in enumerate(MUTANTS):
            if src.count(a) != 1:
                res = f"NOT APPLIED ({src.count(a)} matches)"
            else:
                p = os.path.join(tmp, f"m{k}.py")
                with open(p, "w", encoding="utf-8") as fh:
                    fh.write(src.replace(a, b))
                r = subprocess.run([sys.executable, "-c", TLL.LAUNCH, os.path.abspath(__file__), "one", p], capture_output=True, text=True, env=ENV)
                tail = ((r.stderr or r.stdout).strip().splitlines() or ["no output"])[-1][:150]
                res = "SURVIVED" if r.returncode == 0 else ("killed" if tail.startswith(("AssertionError", "Kill", "jlive_train_under_test.Kill")) else "CRASHED: " + tail)
            print(f"{res[:170]:<12} {name}", flush=True)
            if not res.startswith("killed"):
                left.append(name)
    print(f"{len(MUTANTS) - len(left)} of {len(MUTANTS)} killed", flush=True)
    return left


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        selftest(sys.argv[2])
        sys.exit(0)
    selftest()
    real_check()
    if len(sys.argv) < 2 or sys.argv[1] != "selftests":
        left = run_mutants()
        assert not left, f"mutants survived or crashed: {left}"
    print("test_jlive_train ok")
