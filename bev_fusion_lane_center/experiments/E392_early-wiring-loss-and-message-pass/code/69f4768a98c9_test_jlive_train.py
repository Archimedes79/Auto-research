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
    ("one joint clip over both groups", "nh, nn_, np_ = (torch.nn.utils.clip_grad_norm_([p for p in ps if p.grad is not None], CLIP) for ps in (self.heads, self.nodes, self.pyrs))",
     "nh = nn_ = np_ = torch.nn.utils.clip_grad_norm_([p for p in self.heads + self.nodes + self.pyrs if p.grad is not None], CLIP)"),
    ("the node head is not clipped", "for ps in (self.heads, self.nodes, self.pyrs))", "for ps in (self.heads, [], self.pyrs))"),
    ("the pyramid is not clipped on its own (E393)", "for ps in (self.heads, self.nodes, self.pyrs))", "for ps in (self.heads, self.nodes + self.pyrs, []))"),
    ("fit's joint clip is on", "clip=0, ckpt_every=ckpt_every", "clip=CLIP, ckpt_every=ckpt_every"),
    ("the frozen check is skipped", "            if held:\n", "            if False:\n"),
    ("the learning check is skipped", "            if not (math.isfinite(u) and u > 0):\n", "            if False:\n"),
    ("a LabelOverflow step is not counted", "            m.rec[\"overflow\"].append(i)\n", "            pass\n"),
    ("a LabelOverflow step is trained", "            return torch.full((), float(\"nan\")), dict(", "            return 0.0 * sum(p.sum() for p in m.head.parameters()), dict("),
    ("the overflow kill is off", "    if done >= OVERFLOW_FROM and n > OVERFLOW_MAX * done:\n", "    if False:\n"),
    ("the overflow kill acts before step 100", "    if done >= OVERFLOW_FROM and n > OVERFLOW_MAX * done:\n", "    if n > OVERFLOW_MAX * done:\n"),
    ("the B-rise kill reads the whole curve", "_mean(b[-WIN:]) > B_RISE * _mean(b[:WIN])", "_mean(b) > B_RISE * _mean(b[:WIN])"),
    ("the B-rise kill acts in z", "    if b_rise and done > WIN and", "    if done > WIN and"),
    ("the B-rise kill acts where B''s loss is not in the loss (E393 n / e)", "    b_rise = tn and w > 0 ", "    b_rise = tn "),
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
    ("g_b is the head's gradient (the share inverted)", "gb = torch.autograd.grad(w * loss_b, nodes,", "gb = torch.autograd.grad(loss_h, nodes,"),
    ("an export failure leaves the state done", "        run.status(state=\"failed\", error=repr(e))\n", "        pass\n"),
    ("a kill writes no KILLED file", "        with open(run.path(\"KILLED\"), \"w\", encoding=\"utf-8\") as fh:\n            fh.write(str(e) + \"\\n\")\n", "        pass\n"),
    ("a killed folder is resumed", "    assert not os.path.exists(killed), f\"{folder} was stopped", "    assert True, f\"{folder} was stopped"),
    ("the moving labels of one batch", "collections.deque(maxlen=LOG_EVERY), dict(last=None, checked=False)", "collections.deque(maxlen=1), dict(last=None, checked=False)"),
    ("nn_min counts a token with itself", " & ~torch.eye(v.shape[1], dtype=torch.bool, device=v.device)", ""),
    ("pos_grad_max is never set", "                tk[\"pos\"].register_hook(", "                (lambda f_: None)("),
    ("a head-run option stored as None is kept", "g = lambda k: h[k] if h.get(k) is not None else JKT.ARCH_DEFAULTS.get(k)", "g = lambda k: h.get(k, JKT.ARCH_DEFAULTS.get(k))"),
    ("the head run's rounds are not carried (E392)", "rounds=g(\"rounds\"), wire_weight=", "rounds=0, wire_weight="),
    ("the head run's wire weight is not carried (E392)", "wire_weight=g(\"wire_weight\"), msg=", "wire_weight=0.0, msg="),
    ("the head run's msg is not carried (E392)", "msg=g(\"msg\"),\n", "msg=\"real\",\n"),
    ("the wire weight is not passed to the head's loss (E392)", "pos_loss=args[\"pos_loss\"], wire_w=args[\"wire_weight\"])", "pos_loss=args[\"pos_loss\"])"),
    ("the recipe lacks the E392 options", "\"lr\", \"rounds\", \"wire_weight\", \"msg\")", "\"lr\")"),
    ("a config of before does not resume (E392)", "(c.get(\"args\") or {}).get(k, RECIPE_DEFAULTS.get(k, JKT.ARCH_DEFAULTS.get(k)))", "(c.get(\"args\") or {}).get(k)"),
    ("a config of before E393 does not resume", "RECIPE_DEFAULTS.get(k, JKT.ARCH_DEFAULTS.get(k))", "JKT.ARCH_DEFAULTS.get(k)"),
    ("the recipe lacks the E393 keys", ",\n          \"b_state_from\", \"b_state_sha256\", \"train_pyr\", \"pyr_lr_mult\")", ")"),
    ("the start state is not loaded (E393)", "        src.model.load_state_dict(torch.load(p, map_location=\"cpu\", weights_only=False)[\"b_state\"])\n", "        pass\n"),
    ("the start run's base B' file is not checked (E393)", "        assert s[\"args\"].get(\"b_sha256\") == b_sha,", "        assert True,"),
    ("the start file is not re-checked at the build (E393)", "        assert JKT.file_sha256(p) == args[\"b_state_sha256\"],", "        assert True,"),
    ("the pyramid trains in every arm with a node head (E393)", "train_pyr=bool(args.get(\"train_pyr\")))", "train_pyr=bool(args[\"train_nodes\"]))"),
    ("the pyramid's lr multiplier is not passed (E393)", "dict(params=joint.src.pyr_params(), lr_mult=a.pyr_lr_mult)", "dict(params=joint.src.pyr_params())"),
    ("the pyramid-gradient check is skipped (E393)", "                if not (math.isfinite(n_p) and n_p > 0):\n", "                if False:\n"),
    ("the export lets the pyramid move in every arm (E393)", " and not (args.get(\"train_pyr\") and k.startswith(\"pyr.\"))]", " and not k.startswith(\"pyr.\")]"),
    ("the coverage kill is off (E393)", "    if len(cov) > WIN and", "    if False and"),
    ("the coverage kill reads the whole curve (E393)", "_mean(cov[-WIN:]) < COV_DROP", "_mean(cov) < COV_DROP"),
    ("the token-count kill is off (E393)", "    if len(nv) > WIN and", "    if False and"),
    ("the coverage of a step is not recorded (E393)", "        m.rec[\"cov\"].append(win[-1][2] / win[-1][3] if win[-1][3] > 0 else float(\"nan\"))\n", "        pass\n"),
    ("a skipped step leaves no NaN in cov / nv (E393)", "            for k in (\"b\", \"h\", \"cov\", \"nv\"):\n", "            for k in (\"b\", \"h\"):\n"),
    ("--train-pyr is accepted without a live node head (E393)", "    assert not a.train_pyr or (a.train_nodes and a.live_inputs),", "    assert True,"),
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
    head = JK.JTokenHead()
    head.set_step(12000)
    os.makedirs(os.path.join(tmp, "h"))
    torch.save(dict(state=head.state_dict(), args=dict(seed=0, ident="e292_a", none_weight=0.25, head_seed=2, teacher="none"), steps=12000), os.path.join(tmp, "h", "model.pt"))
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
    headr = JK.JTokenHead(rounds=1)                                                         # E392: a rounds head run "hr" (live messages: the zero start moved), early-read weight 0.5
    headr.set_step(12000)
    with torch.no_grad():
        headr.rounds[0].mlp[2].weight.normal_(std=0.1)
    os.makedirs(os.path.join(tmp, "hr"))
    torch.save(dict(state=headr.state_dict(), args=dict(seed=0, ident="e292_a", none_weight=0.25, head_seed=2, rounds=1, wire_weight=0.5, msg="uniform"), steps=12000), os.path.join(tmp, "hr", "model.pt"))
    headn = JK.JTokenHead(rounds=1, msg="none")                                             # E392: a msg none head run "hnx" (deep supervision, the champion's parameters), early-read weight 0.5
    headn.set_step(12000)
    os.makedirs(os.path.join(tmp, "hnx"))
    torch.save(dict(state=headn.state_dict(), args=dict(seed=0, ident="e292_a", none_weight=0.25, head_seed=2, rounds=1, wire_weight=0.5, msg="none"), steps=12000), os.path.join(tmp, "hnx", "model.pt"))
    heads = JK.JTokenHead(rounds=2, msg="self")                                             # E392: a msg self head run "hsx" (the capacity null: the rounds on each token's own features), early-read weight 0.5
    heads.set_step(12000)
    os.makedirs(os.path.join(tmp, "hsx"))
    torch.save(dict(state=heads.state_dict(), args=dict(seed=0, ident="e292_a", none_weight=0.25, head_seed=2, rounds=2, wire_weight=0.5, msg="self"), steps=12000), os.path.join(tmp, "hsx", "model.pt"))
    import jtoken_train as JKT
    st = {k: v.clone() for k, v in m.state_dict().items()}                                  # E393: a start run "st" (a jlive model.pt) whose b_state differs from the B' file in the node head AND the pyramid
    with torch.no_grad():
        for k in st:
            if (k.startswith("nodes.") or k.startswith("pyr.")) and st[k].is_floating_point() and st[k].dim() > 1:
                st[k] += 0.01 * torch.randn(st[k].shape, generator=g)
    os.makedirs(os.path.join(tmp, "st"))
    torch.save(dict(state=head.state_dict(), args=dict(seed=0, b_sha256=JKT.file_sha256(os.path.join(tmp, "b", "model.pt"))), b_state=st, b_args=dict(width=16, soft_nms=SOFT)), os.path.join(tmp, "st", "model.pt"))
    return dict(b_dir=os.path.join(tmp, "b"), A=LLT.collate(items[:2]), B=LLT.collate(items[2:]), st=st)


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


def direct(T, w, tmp, train_nodes, live=1, steps=1, head="h"):
    """the Joint, the loss of one step and the hook of an arm, built as `train` builds them (no fit); --steps 1: step 1 is the last, a logged and (s4) a share step"""
    args = T.resolve(T.parse_args(cli(w, "d", train_nodes, live, steps=steps, extra=["--head-run", head])), tmp)
    joint = T.build(args, tmp)[0]
    run = NoRun()
    hook = T.StepHook(run, joint)
    joint.train()
    return joint, T.make_loss(run, joint, args, hook), hook, run, args


def rounds_head(T, w, tmp):
    """E392: the stage-2 trainer carries the head run's rounds / msg / wire weight. `resolve` hands them on (a head run without them: the defaults), the head is built with them and loaded strictly, the step's loss is the head's loss with the early
    reads (info nll1_s, nll1_l) + B''s, the recipe and model.pt's args hold them, `load_model` rebuilds the head, a config of before (without them) resumes, a rounds head trains one step through `train` (the log row carries nll1_s)"""
    import numpy as np
    import torch
    import jtoken as JK
    import jtoken_train as JKT
    import lane_live_tok as LLT
    B = w["B"]
    a0, a1 = (T.resolve(T.parse_args(cli(w, "d", 1, 1, steps=1, extra=["--head-run", h])), tmp) for h in ("h", "hr"))
    assert (a0["rounds"], a0["wire_weight"], a0["msg"]) == (0, 0.0, "real") and (a1["rounds"], a1["wire_weight"], a1["msg"]) == (1, 0.5, "uniform"), (a0, a1)
    joint, lf, hook, run, args = direct(T, w, tmp, 1, 1, head="hr")
    assert len(joint.head.rounds) == 1 and joint.head.msg == "uniform" and joint.head.live_inputs
    ref = copy.deepcopy(joint)
    loss, info = lf(joint, B)
    assert "nll1_s" in info and "nll1_l" in info and "nll2_s" not in info, sorted(info)
    no, tk = ref.src.tokens(B["x"])
    lbatch = ref.src.labelled(tk, B["table"], B["token"], B["row"], gate=False)
    out_r = ref.head(LLT.head_inputs(tk))
    lh, ih = JK.jtoken_loss(out_r, lbatch, none_weight=0.25, pos_loss="sum", wire_w=0.5)
    lh0 = JK.jtoken_loss(out_r, lbatch, none_weight=0.25, pos_loss="sum")[0]
    lb, _ = LLT.b_loss(no, B["targets"])
    assert float(lh) > float(lh0) + 1e-3 and abs(float(loss) - float(lh + lb)) < 1e-6 and abs(info["nll1_s"] - ih["nll1_s"]) < 1e-6 and abs(info["head"] - float(lh)) < 1e-6, "the head run's wire weight is not in the loss of the step"
    with Built(T, [B]) as bt:
        T.train(T.parse_args(cli(w, "rr", 1, steps=1, extra=["--head-run", "hr"])), runs=tmp, ckpt_every=500)
    last = [json.loads(x) for x in open(os.path.join(tmp, "rr", "log.jsonl"))][-1]
    assert math.isfinite(last["nll1_s"]) and math.isfinite(last["nll1_l"]) and math.isfinite(last["head"]), last
    cfg = json.load(open(os.path.join(tmp, "rr", "config.json")))["args"]
    assert (cfg["rounds"], cfg["wire_weight"], cfg["msg"]) == (1, 0.5, "uniform")
    ck = torch.load(os.path.join(tmp, "rr", "model.pt"), weights_only=False)
    assert (ck["args"]["rounds"], ck["args"]["wire_weight"], ck["args"]["msg"]) == (1, 0.5, "uniform")
    hm, _ = JKT.load_model("rr", "cpu", tmp)
    assert len(hm.rounds) == 1 and hm.msg == "uniform" and hm.live_inputs and all(torch.equal(hm.state_dict()[k], bt.joints[0].head.state_dict()[k].cpu()) for k in hm.state_dict()), "load_model does not rebuild the trained rounds head"
    h0 = torch.load(os.path.join(tmp, "hr", "model.pt"))["state"]
    assert any(not torch.equal(hm.state_dict()[k], h0[k]) for k in h0 if k.startswith("rounds.")), "the rounds head did not train"
    assert JKT.build_head(ck["args"]).state_dict().keys() == hm.state_dict().keys()
    folder = os.path.join(tmp, "rr")
    T.check_start(cfg, folder)                                                              # the same recipe resumes; the head run's options are in the recipe
    for k, v in (("rounds", 2), ("wire_weight", 0.25), ("msg", "real")):
        refuses(lambda: T.check_start(dict(cfg, **{k: v}), folder), "of another run")
    os.makedirs(os.path.join(tmp, "oldcfg"))                                               # a config of before the options (no rounds / wire_weight / msg) resumes as the defaults
    json.dump(dict(design="jlive", args={k: v for k, v in cfg.items() if k not in ("rounds", "wire_weight", "msg")}), open(os.path.join(tmp, "oldcfg", "config.json"), "w"))
    T.check_start(dict(cfg, rounds=0, wire_weight=0.0, msg="real"), os.path.join(tmp, "oldcfg"))
    refuses(lambda: T.check_start(cfg, os.path.join(tmp, "oldcfg")), "of another run")
    print(f"rounds head through jlive_train: resolve carries rounds / wire weight / msg from the head run (a head run without them: 0 / 0 / real), the step's loss = the head's loss with the early read ({float(lh):.4f} against {float(lh0):.4f} without) + B''s, "
          f"one step trains (nll1_s {last['nll1_s']:.4f} in the log row), the three options in the recipe, config and model.pt, load_model rebuilds the head, a config of before resumes as the defaults", flush=True)


def none_head(T, w, tmp):
    """E392 msg none (deep supervision without exchange) through the stage-2 trainer: `resolve` carries rounds 1 / wire weight 0.5 / msg none of the head run, the head is built with no round module and the champion's parameters (the strict load of the run's
    state), it trains one step through `train` (in training mode: the log row carries nll1_s of the early read), model.pt's args hold msg none, `load_model` rebuilds the champion-shaped head whose eval forward has no early read"""
    import torch
    import jtoken as JK
    import jtoken_train as JKT
    a = T.resolve(T.parse_args(cli(w, "d", 1, 1, steps=1, extra=["--head-run", "hnx"])), tmp)
    assert (a["rounds"], a["wire_weight"], a["msg"]) == (1, 0.5, "none"), a
    with Built(T, [w["B"]]) as bt:
        T.train(T.parse_args(cli(w, "nx", 1, steps=1, extra=["--head-run", "hnx"])), runs=tmp, ckpt_every=500)
    head = bt.joints[0].head
    assert len(head.rounds) == 0 and head.n_early == 1 and head.msg == "none" and JK.count_params(head) == 127131, "the head of a msg none run is not the champion-shaped head with one early read"
    last = [json.loads(x) for x in open(os.path.join(tmp, "nx", "log.jsonl"))][-1]
    assert math.isfinite(last["nll1_s"]) and math.isfinite(last["nll1_l"]) and math.isfinite(last["head"]) and "nll2_s" not in last, last
    ck = torch.load(os.path.join(tmp, "nx", "model.pt"), weights_only=False)
    assert (ck["args"]["rounds"], ck["args"]["wire_weight"], ck["args"]["msg"]) == (1, 0.5, "none")
    hm, _ = JKT.load_model("nx", "cpu", tmp)
    h0 = torch.load(os.path.join(tmp, "hnx", "model.pt"))["state"]
    assert len(hm.rounds) == 0 and hm.n_early == 1 and hm.msg == "none" and list(hm.state_dict()) == list(JK.JTokenHead().state_dict()) and not hm.training, "load_model does not rebuild the msg none head"
    assert any(not torch.equal(hm.state_dict()[k], h0[k]) for k in h0 if k != "step"), "the msg none head did not train"
    with torch.no_grad():
        assert "logp_rounds" not in hm(JK.synthetic_batch(2, 48, seed=0)), "the msg none head reads early in eval mode"
    print(f"msg none head through jlive_train: resolve carries rounds / wire weight / msg none from the head run, the champion-shaped head (127131 parameters, no round module) trains one step in training mode (nll1_s {last['nll1_s']:.4f} in the log row), "
          f"model.pt holds msg none, load_model rebuilds it, its eval forward has no early read", flush=True)


def self_head(T, w, tmp):
    """E392 msg self (the capacity null) through the stage-2 trainer: `resolve` carries rounds 2 / wire weight 0.5 / msg self of the head run, the head is built with the two round modules (168603 parameters, the strict load of the run's state), it trains
    one step through `train` (the log row carries nll1_s and nll2_s), model.pt's args hold msg self, `load_model` rebuilds the head whose eval forward has no early read and whose rounds act"""
    import torch
    import jtoken as JK
    import jtoken_train as JKT
    a = T.resolve(T.parse_args(cli(w, "d", 1, 1, steps=1, extra=["--head-run", "hsx"])), tmp)
    assert (a["rounds"], a["wire_weight"], a["msg"]) == (2, 0.5, "self"), a
    with Built(T, [w["B"]]) as bt:
        T.train(T.parse_args(cli(w, "sx", 1, steps=1, extra=["--head-run", "hsx"])), runs=tmp, ckpt_every=500)
    head = bt.joints[0].head
    assert len(head.rounds) == 2 and head.n_early == 2 and head.msg == "self" and JK.count_params(head) == 168603, "the head of a msg self run is not the two-round head"
    last = [json.loads(x) for x in open(os.path.join(tmp, "sx", "log.jsonl"))][-1]
    assert all(math.isfinite(last[k]) for k in ("nll1_s", "nll1_l", "nll2_s", "nll2_l", "head")), last
    ck = torch.load(os.path.join(tmp, "sx", "model.pt"), weights_only=False)
    assert (ck["args"]["rounds"], ck["args"]["wire_weight"], ck["args"]["msg"]) == (2, 0.5, "self")
    hm, _ = JKT.load_model("sx", "cpu", tmp)
    assert len(hm.rounds) == 2 and hm.msg == "self" and not hm.training and all(torch.equal(hm.state_dict()[k], head.state_dict()[k].cpu()) for k in hm.state_dict()), "load_model does not rebuild the msg self head"
    with torch.no_grad():
        assert "logp_rounds" not in hm(JK.synthetic_batch(2, 48, seed=0)), "the msg self head reads early in eval mode"
    print(f"msg self head through jlive_train: resolve carries rounds 2 / wire weight 0.5 / msg self from the head run, the two-round head (168603 parameters) trains one step (nll1_s {last['nll1_s']:.4f}, nll2_s {last['nll2_s']:.4f} in the log row), "
          f"model.pt holds msg self, load_model rebuilds it, its eval forward has no early read", flush=True)


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
    # E393: the coverage and token-count kills (every arm, any 200 steps after step 200 against steps 1 - 200; NaN left out); the B-rise kill only where B's loss is in the loss
    cv = lambda cov, nv=None: dict(rec([1.0] * len(cov)), cov=list(cov), nv=list(nv if nv is not None else [100.0] * len(cov)))  # noqa: E731
    for br in (True, False):
        assert T.verdict(cv([0.98] * 200 + [0.932] * 200), br) is None and says(T.verdict(cv([0.98] * 200 + [0.93] * 200), br), "coverage fell"), "coverage: below 0.95 x the start mean"
        assert says(T.verdict(cv([0.98] * 200 + [0.98] * 150 + [0.5] * 50), br), "coverage fell") and T.verdict(cv([0.98] * 199 + [0.5]), br) is None, "coverage: the last 200 steps, after step 200"
        assert says(T.verdict(cv([0.98] * 199 + [nan] + [0.9] * 199 + [nan]), br), "coverage fell"), "coverage: a NaN step is left out"
        assert T.verdict(cv([0.9] * 400, [100.0] * 200 + [124.9] * 200), br) is None and says(T.verdict(cv([0.9] * 400, [100.0] * 200 + [125.1] * 200), br), "token set blew up"), "token count: above 1.25 x the start mean"
        assert T.verdict(cv([0.9] * 400, [100.0] * 200 + [50.0] * 200), br) is None, "token count: a fall is no kill"
    assert T.verdict(dict(rec([1.0] * 200 + [2.0] * 50)), False) is None and T.verdict(rec([1.0] * 250), True) is None, "a record of before cov / nv (no keys): judged without them"
    assert T.COV_DROP == 0.95 and T.NV_RISE == 1.25
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
        lbatch = ref.src.labelled(tk, B["table"], B["token"], B["row"], gate=False)
        lh, _ = JK.jtoken_loss(ref.head(LLT.head_inputs(tk)), lbatch, none_weight=0.25, pos_loss="sum")
        lb, _ = LLT.b_loss(no, B["targets"])
        want = lh + lb if tn else lh
        assert abs(float(loss) - float(want)) < 1e-6 and abs(info["head"] - float(lh)) < 1e-6 and abs(info["b_loss"] - float(lb)) < 1e-6, f"arm {T.arm(tn, live)}: the loss of a step is not the head's loss {'+ B''s' if tn else 'alone'} ({float(loss)} {float(want)})"
        assert loss.requires_grad and (not tn or not live or any("> 0: s4" in x for x in run.lines)) and (live or any("0: arm b" in x for x in run.lines)), (T.arm(tn, live), run.lines)
        assert hook.cur["i"] == 1 and hook.cur["info"] is info and hook.cur["logged"] and any("step 1 frames ['f2', 'f3']" in x for x in run.lines)
        if tn and live:                                                                     # the share step: the gradient handed to the hook is B''s (b_weight 1) on the node head, not the head's
            want_gb = torch.autograd.grad(lb, ref.src.trainable(), retain_graph=True, allow_unused=True)
            assert hook.cur["gb"] is not None and len(hook.cur["gb"]) == len(want_gb), "s4: no g_b at the last step (a share step)"
            assert all((x is None and y is None) or torch.allclose(x, y, rtol=1e-5, atol=1e-7) for x, y in zip(hook.cur["gb"], want_gb)), "the hook's g_b is not the gradient of B''s loss"
        else:
            assert hook.cur["gb"] is None
        # the hazards of the step: nn_min by hand (exact pairwise distances); pos_grad_max NaN until the backward, then the head's largest |gradient| on the positions (s4; NaN in z and b)
        want_nn = min(float(torch.cdist(p_, p_, compute_mode="donot_use_mm_for_euclid_dist").add(torch.eye(len(p_), dtype=torch.float64) * 1e9).min()) for p_ in (tk["pos"][f][tk["valid"][f].bool()].detach().double() for f in range(2)))
        assert abs(info["nn_min"] - want_nn) < 1e-9 and want_nn > 0 and math.isnan(info["pos_grad_max"]), (info["nn_min"], want_nn)
        want_pg = float(torch.autograd.grad(lh, tk["pos"], retain_graph=True)[0].abs().max()) if tn and live else float("nan")
        loss.backward()
        assert abs(info["pos_grad_max"] - want_pg) < 1e-5 * max(1.0, want_pg) if tn and live else math.isnan(info["pos_grad_max"]), (T.arm(tn, live), info["pos_grad_max"], want_pg)
        # the moving labels over the last 100 steps of the process: B, then A (no GT; A alone would give pair_cov = gt_cov = 0)
        lab = lbatch["lab"]
        lab_a = ref.src.labelled(ref.src.tokens(w["A"]["x"])[1], w["A"]["table"], w["A"]["token"], w["A"]["row"], gate=False)["lab"]
        info2 = lf(joint, w["A"])[1]
        assert info2["gt_cov"] == sum(float(x["n_gt_assigned"]) for x in lab + lab_a) / max(sum(float(x["n_gt"]) for x in lab + lab_a), 1.0) > 0 and info2["pair_cov"] == info["pos_pairs"] / max(info["gt_links"], 1.0) > 0, (info2, info)
        assert info2["n_valid"] == float(np.mean([x["n_valid"] for x in lab + lab_a])) and info["overflow"] == 0.0
        assert len(joint.rec["b"]) == 2 and abs(joint.rec["h"][0] - float(lh)) < 1e-6
        cov_b = sum(float(x["n_gt_assigned"]) for x in lab) / sum(float(x["n_gt"]) for x in lab)
        assert joint.rec["cov"][0] == cov_b > 0 and math.isnan(joint.rec["cov"][1]) and joint.rec["nv"] == [float(np.mean([x["n_valid"] for x in lab])), float(np.mean([x["n_valid"] for x in lab_a]))], (joint.rec["cov"], joint.rec["nv"])   # A has no GT: NaN
    # a head-run option stored as None means the default (as `jtoken_train.arch` reads it)
    ck = torch.load(os.path.join(tmp, "h", "model.pt"), weights_only=False)
    os.makedirs(os.path.join(tmp, "hn"))
    torch.save(dict(ck, args=dict(ck["args"], none_weight=None, pos_loss=None)), os.path.join(tmp, "hn", "model.pt"))
    an = T.resolve(T.parse_args(["--name", "x", "--seed", "0", "--head-run", "hn", "--b-run", w["b_dir"], "--train-nodes", "1"]), tmp)
    assert an["none_weight"] == 1.0 and an["pos_loss"] == "sum", an
    # a head run of an option removed in D117 (E377's near weight, the c-rule teacher) is refused, not trained without it (`check_removed`)
    for n_, extra in (("hr1", dict(none_weight_near=0.73)), ("hr2", dict(teacher="rule"))):
        os.makedirs(os.path.join(tmp, n_))
        torch.save(dict(ck, args=dict(ck["args"], **extra)), os.path.join(tmp, n_, "model.pt"))
        refuses(lambda: T.resolve(T.parse_args(["--name", "x", "--seed", "0", "--head-run", n_, "--b-run", w["b_dir"], "--train-nodes", "1"]), tmp), "options removed in D117")
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
        joint.rec = dict(b=[1.0] * 200, h=[1.0] * 200, overflow=[7], cov=[], nv=[])
        assert math.isnan(float(lf(joint, B)[0])) and joint.rec["overflow"] == [7, 201], "2 of 201 steps: no kill"
        joint.rec = dict(b=[1.0] * 100, h=[1.0] * 100, overflow=[7], cov=[], nv=[])
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
    joint.rec = dict(b=[1.0] * 399, h=[1.0] * 399, overflow=[], cov=[], nv=[])
    kills(lambda: lf(joint, B), "B''s node loss rose")
    joint, lf, *_ = direct(T, w, tmp, 0, 1)
    joint.rec = dict(b=[1.0] * 399, h=[1.0] * 399, overflow=[], cov=[], nv=[])
    lf(joint, B)
    # the flags: the first step of a process after step 5 is a resume, a step number that does not grow a roll-back (both in status)
    joint, lf, _, run, _ = direct(T, w, tmp, 0, 1)
    joint.rec = dict(b=[1.0] * 5, h=[1.0] * 5, overflow=[], cov=[], nv=[])
    lf(joint, B)
    joint.rec = dict(b=[1.0] * 5, h=[1.0] * 5, overflow=[], cov=[], nv=[])
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
    assert 0 < last["ushare"] < 1 and 0 < last["gshare"] < 0.1 and last["upd_head"] > 0 and last["overflow"] == 0, last                    # the toy's B loss dominates the node head's gradient
    assert last["nn_min"] > 0 and last["pos_grad_max"] > 0 and math.isfinite(last["pos_grad_max"]), last
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
    sz = json.load(open(os.path.join(tmp, "z", "status.json")))
    assert os.path.exists(os.path.join(tmp, "z", ".done.FALSE")) and not os.path.exists(os.path.join(tmp, "z", ".done")) and sz["state"] == "failed" and "must stay as at the start" in sz["error"], sz
    # a kill survives a resume: the reason in KILLED and status.json, the folder is refused at the next start
    ob = LLT.b_loss
    LLT.b_loss = lambda no, tg: (torch.tensor(float("nan")), {})
    try:
        with Built(T, [w["B"]]):
            kills(lambda: T.train(T.parse_args(cli(w, "kd", 1)), runs=tmp, ckpt_every=500), "not finite")
    finally:
        LLT.b_loss = ob
    assert os.path.exists(os.path.join(tmp, "kd", "KILLED")), "a kill wrote no KILLED file"
    assert "not finite" in open(os.path.join(tmp, "kd", "KILLED")).read() and "not finite" in json.load(open(os.path.join(tmp, "kd", "status.json")))["killed"]
    with Built(T, [w["B"]]):
        refuses(lambda: T.train(T.parse_args(cli(w, "kd", 1)), runs=tmp, ckpt_every=500), "stopped by a kill rule")
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
              none_weight=lambda v: v * 2, pos_loss=lambda v: "max", head_seed=lambda v: v + 1, bs=lambda v: v + 1, steps=lambda v: v + 1, lr=lambda v: v * 2, rounds=lambda v: v + 1, wire_weight=lambda v: v + 0.5, msg=lambda v: "uniform",
              b_state_from=lambda v: "st", b_state_sha256=lambda v: "0" * 64, train_pyr=lambda v: 1 - v, pyr_lr_mult=lambda v: v * 2)
    assert set(ch) == set(T.RECIPE), f"a recipe key without a refusal test: {set(ch) ^ set(T.RECIPE)}"
    for k in ch:
        refuses(lambda: T.check_start(dict(cfg, **{k: ch[k](cfg[k])}), folder), "of another run")


def pyr_arms(T, w, tmp):
    """E393 through the trainer: the CLI refusals; --b-state-from loads the start run's b_state bit for bit (its file sha in the recipe; another base B' file, no b_state, a changed file refused); arm p (pyramid trainable at lr x 0.1, the node anchor
    on) two steps through `train` between spy hooks: two AdamW groups, the pyramid's rate 0.1 x the head's in every step, the three groups clipped each on its own, the pyramid and the node head moved from the START state, the logged numbers, the
    log lines; c (pyramid frozen) and e (b_weight 0) from the same start: c moves only the node head; the pyramid-gradient kill, the frozen pyramid in c, the B-rise kill only with the anchor; p resumed = p straight, bit for bit"""
    import torch
    import jtoken_train as JKT
    import trainer
    st, B = w["st"], w["B"]
    rd = lambda n: [json.loads(x) for x in open(os.path.join(tmp, n, "log.jsonl"))]  # noqa: E731
    text = lambda n: open(os.path.join(tmp, n, "train.log"), encoding="utf-8").read()  # noqa: E731
    P = ["--b-state-from", "st", "--train-pyr", "1", "--pyr-lr-mult", "0.1"]
    refuses(lambda: T.parse_args(cli(w, "x", 0, extra=["--train-pyr", "1"])), "needs --train-nodes 1")
    refuses(lambda: T.parse_args(cli(w, "x", 1, live=0, extra=["--train-pyr", "1"])), "needs --train-nodes 1")
    refuses(lambda: T.parse_args(cli(w, "x", 1, extra=["--pyr-lr-mult", "0.1"])), "only with --train-pyr 1")
    # the start state
    a = T.resolve(T.parse_args(cli(w, "x", 1, extra=["--b-state-from", "st"])), tmp)
    assert a["b_state_from"] == "st" and a["b_state_sha256"] == JKT.file_sha256(os.path.join(tmp, "st", "model.pt")) and a["train_pyr"] == 0 and a["pyr_lr_mult"] == 1.0, a
    j = T.build(a, tmp)[0]
    assert all(torch.equal(j.src.model.state_dict()[k], st[k]) for k in st), "the start B' is not the b_state of --b-state-from"
    a0 = T.resolve(T.parse_args(cli(w, "x", 1)), tmp)
    assert a0["b_state_from"] is None and a0["b_state_sha256"] is None and not all(torch.equal(T.build(a0, tmp)[0].src.model.state_dict()[k], st[k]) for k in st), "without --b-state-from: the B' file's state"
    ck = torch.load(os.path.join(tmp, "st", "model.pt"), weights_only=False)
    for n_, c_ in (("st_other", dict(ck, args=dict(ck["args"], b_sha256="0" * 64))), ("st_none", {k: v for k, v in ck.items() if k != "b_state"})):
        os.makedirs(os.path.join(tmp, n_))
        torch.save(c_, os.path.join(tmp, n_, "model.pt"))
    refuses(lambda: T.resolve(T.parse_args(cli(w, "x", 1, extra=["--b-state-from", "st_other"])), tmp), "is not --b-run")
    refuses(lambda: T.resolve(T.parse_args(cli(w, "x", 1, extra=["--b-state-from", "st_none"])), tmp), "carries no b_state")
    refuses(lambda: T.build(dict(a, b_state_sha256="0" * 64), tmp), "changed since")
    # arm p through `train`, between spy hooks (as `runs`)
    seen, real_fit = dict(before=[], after=[], lr=[]), trainer.fit
    with Built(T, [B]) as bt:
        state = {}
        hb = register_optimizer_step_pre_hook(lambda opt, a_, k_: seen["before"].append(state["n"]()) if "n" in state else None)

        def fit(model, loss_fn, run, **kw):
            groups = dict(head=list(model.head.parameters()), nodes=model.src.node_params(), pyr=model.src.pyr_params())
            state["n"] = lambda: {g: math.sqrt(sum(float(p.grad.double().square().sum()) for p in ps if p.grad is not None)) for g, ps in groups.items()}

            def after(opt, a_, k_):
                seen["after"].append(state["n"]())
                seen["lr"].append([g["lr"] for g in opt.param_groups])
            ha = register_optimizer_step_pre_hook(after)
            try:
                seen["fit_kw"] = kw
                return real_fit(model, loss_fn, run, **kw)
            finally:
                ha.remove()
        T.T.fit = fit
        try:
            T.train(T.parse_args(cli(w, "p", 1, extra=P)), runs=tmp, ckpt_every=500)
        finally:
            T.T.fit = real_fit
            hb.remove()
        jp = bt.joints[0]
    assert len(seen["before"]) == len(seen["after"]) == 2 and all(bf["pyr"] > 0 for bf in seen["before"]), seen["before"]
    for bf, af in zip(seen["before"], seen["after"]):
        for g in ("head", "nodes", "pyr"):
            assert abs(af[g] - min(bf[g], T.CLIP * bf[g] / (bf[g] + 1e-6))) < 1e-4 * max(bf[g], 1e-3), f"the {g} group is not clipped on its own: before {bf} after {af}"
    assert any(bf["pyr"] > T.CLIP or bf["nodes"] > T.CLIP for bf in seen["before"]), f"the test needs an active clip on a B' group: {seen['before']}"
    assert all(len(l_) == 2 and l_[0] > 0 and abs(l_[1] - 0.1 * l_[0]) < 1e-15 for l_ in seen["lr"]), f"the pyramid's rate is not 0.1 x the head's: {seen['lr']}"
    g0, g1 = seen["fit_kw"]["params"]
    assert [id(p) for p in g0["params"]] == [id(p) for p in list(jp.head.parameters()) + jp.src.node_params()] and [id(p) for p in g1["params"]] == [id(p) for p in jp.src.pyr_params()] and g1["lr_mult"] == 0.1 and "lr_mult" not in g0
    last = rd("p")[-1]
    for k in ("gn_head", "gn_nodes", "gn_pyr", "pshare", "clip_pyr", "upd_nodes", "upd_pyr", "upd_head", "cov_mean", "nv_mean", "b_loss"):
        assert k in last and math.isfinite(last[k]), (k, last)
    assert last["gn_pyr"] > 0 and 0 < last["pshare"] < 1 and last["upd_pyr"] > 0 and abs(last["gn_pyr"] - seen["before"][-1]["pyr"]) < 1e-4 * last["gn_pyr"], last
    tx = text("p")
    assert "jlive arm s4 + pyramid" in tx and "its state from st" in tx and "lr x 0.1" in tx and "gives B''s pyramid a gradient of norm" in tx and "'lines05.weight', 'lines05.bias'" in tx, tx[-2000:]
    cfg = json.load(open(os.path.join(tmp, "p", "config.json")))["args"]
    assert cfg["train_pyr"] == 1 and cfg["pyr_lr_mult"] == 0.1 and cfg["b_state_from"] == "st"
    T.check_start(cfg, os.path.join(tmp, "p"))

    def moved(run):                                                                         # against the START state (st), not the B' file
        b = T.load_b(run, "cpu", tmp).state_dict()
        return sorted({k.split(".")[0] for k in b if not torch.equal(b[k], st[k])})
    assert moved("p") == ["nodes", "pyr"] and all(torch.equal(T.load_b("p", "cpu", tmp).state_dict()[k], jp.src.model.state_dict()[k]) for k in st), moved("p")
    with Built(T, [B]):
        T.train(T.parse_args(cli(w, "c", 1, extra=["--b-state-from", "st"])), runs=tmp, ckpt_every=500)
        T.train(T.parse_args(cli(w, "e", 1, extra=P + ["--b-weight", "0"])), runs=tmp, ckpt_every=500)
    assert moved("c") == ["nodes"] and moved("e") == ["nodes", "pyr"], (moved("c"), moved("e"))
    assert "gives B''s pyramid" not in text("c") and "gn_pyr" not in rd("c")[-1] and "gn_pyr" in rd("e")[-1]
    # the kills: the pyramid without a gradient (its outputs cut), a pyramid gradient in c, the B-rise kill with the anchor only

    def built(extra):
        args = T.resolve(T.parse_args(cli(w, "d", 1, steps=1, extra=extra)), tmp)
        joint = T.build(args, tmp)[0]
        run = NoRun()
        hook = T.StepHook(run, joint)
        joint.train()
        return joint, T.make_loss(run, joint, args, hook), hook, run
    joint, lf, _, _ = built(P)
    fw = joint.src.model.pyr.forward
    joint.src.model.pyr.forward = lambda x: {k: v.detach() for k, v in fw(x).items()}
    kills(lambda: lf(joint, B), "gives B''s pyramid no gradient")
    joint, lf, hook, _ = built(["--b-state-from", "st"])
    opt = torch.optim.AdamW(list(joint.head.parameters()), lr=1e-4)
    for p in joint.head.parameters():
        p.grad = torch.ones_like(p)
    pp = next(joint.src.model.pyr.parameters())
    pp.grad = torch.ones_like(pp)
    kills(lambda: hook(opt, (), {}), "hold a gradient")
    for extra, killed in ((P, True), (P + ["--b-weight", "0"], False)):
        joint, lf, _, _ = built(extra)
        joint.rec = dict(b=[1.0] * 399, h=[1.0] * 399, overflow=[], cov=[], nv=[])
        if killed:
            kills(lambda: lf(joint, B), "B''s node loss rose")
        else:
            lf(joint, B)
    # p resumed after a crash at step 3 = p straight, bit for bit (the pyramid's group and its rate survive the checkpoint)
    with Built(T, [B]):
        T.train(T.parse_args(cli(w, "pst", 1, steps=4, extra=P)), runs=tmp, ckpt_every=2)
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
                T.train(T.parse_args(cli(w, "prs", 1, steps=4, extra=P)), runs=tmp, ckpt_every=2)
                raise AssertionError("the run was not killed")
            except RuntimeError as e:
                assert "killed" in str(e), e
            T.make_loss = real_make
            T.train(T.parse_args(cli(w, "prs", 1, steps=4, extra=P)), runs=tmp, ckpt_every=2)
        finally:
            T.make_loss = real_make
    a_, b_ = (torch.load(os.path.join(tmp, n_, "model.pt"), weights_only=False) for n_ in ("pst", "prs"))
    for k in ("state", "b_state"):
        assert all(torch.equal(a_[k][x], b_[k][x]) for x in a_[k]), f"p resumed is not p straight ({k})"
    assert a_["rec"] == b_["rec"] and len(a_["rec"]["cov"]) == 4, (a_["rec"], b_["rec"])
    print("E393 through jlive_train: --b-state-from = the start run's b_state bit for bit (refusals: another base file, no b_state, a changed file); arm p: two groups, the pyramid at 0.1 x the rate in every step, three clip groups, "
          f"pyramid + node head moved from the start, gn_pyr {last['gn_pyr']:.3g} (share {last['pshare']:.3f}); c moves the node head only, e both; the pyramid-gradient kill, the frozen pyramid in c, the B-rise kill with the anchor only; p resumed = p straight", flush=True)


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
        rounds_head(T, w, tmp)
        none_head(T, w, tmp)
        self_head(T, w, tmp)
        runs(T, w, tmp)
        pyr_arms(T, w, tmp)
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
