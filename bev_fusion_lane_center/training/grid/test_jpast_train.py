"""Tests of the E368 trainer (`train/jpast_train.py`) and the read-out (`evaluation/jpast_eval.py`).

    wsl bash tools/env/_py14.sh test_jpast_train.py            # the selftest (a stub head with the S2 interface, and the real `jtoken.JTokenHead` when it has the hook), then the hand mutants of both files: every mutant must make it FAIL
    wsl bash tools/env/_py14.sh test_jpast_train.py selftests  # the selftest only

The loop by hand: the head's step counter held for the forwards of a step and advanced once, the past of step k = `jpast.make_past` of the frames and the poses of step k - 1 (the key weight before and after the switch), the state in the graph (the gradient of the last frame reaches
the first), no GT key in the state, a sample without a previous keyframe cold, arm z = the mean of the frames' losses, the zero-initialised block = arm z, `u` logged, the smoke run with its resume, the hook check. The read-out: the modes stream / truncated / off / still / shuffled against a hand loop,
the donor plan, the cold starts, the dump of the cache frames of VAL (a toy untrained head on 12 real frames: the writer of `jtoken_eval` accepts it). A mutant = one textual change of the source of `jpast_train.py` or `jpast_eval.py`; the mutated copy is loaded in place of the file, four at a time.
"""
import argparse
import concurrent.futures as cf
import copy
import importlib.util
import inspect
import os
import subprocess
import sys
import tempfile

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAIN, EVAL = os.path.join(ROOT, "training", "train", "jpast_train.py"), os.path.join(ROOT, "evaluation", "jpast_eval.py")
ENV = dict(os.environ, BFLC_ROOT=ROOT, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")      # a shared box: one thread per test process
for sub_ in ("training", "training/grid", "training/data", "training/data/prep", "training/train", "evaluation"):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))

T0 = "jpast_train.py"
MUTANTS = [
    (T0, "the head's counter advances per frame", "    m.set_step(s0 + 1)\n    return sum(losses)", "    return sum(losses)"),
    (T0, "the forwards of a step see different counters", "        m.set_step(s0)\n        if no_past:", "        if no_past:"),
    (T0, "the switch is judged on the counter inside the step", "    learned = s0 >= JK.SWITCH", "    learned = m._step_py >= JK.SWITCH"),
    (T0, "the pose of the wrong step", "frames[k - 1], fr, R[:, k], t[:, k], learned, hp[:, k])", "frames[k - 1], fr, R[:, k - 1], t[:, k - 1], learned, hp[:, k])"),
    (T0, "the pose is the inverse (the frames swapped)", "JP.make_past(out_prev, frames[k - 1], fr, R[:, k], t[:, k], learned, hp[:, k])", "JP.make_past(out_prev, fr, frames[k - 1], R[:, k], t[:, k], learned, hp[:, k])"),
    (T0, "the state of the previous step is detached", "        out_prev = out\n", "        out_prev = dict(out, x=out['x'].detach(), w=out['w'].detach())\n"),
    (T0, "the first step reads a past", "past = {} if k == 0 else", "past = {} if k == 99 else"),
    (T0, "has_prev is ignored", "learned, hp[:, k])", "learned)"),
    (T0, "the loss is the sum of the frames", "return sum(losses) / len(losses), infos", "return sum(losses), infos"),
    (T0, "the loss is that of the last frame only", "return sum(losses) / len(losses), infos", "return losses[-1], infos"),
    (T0, "arm z reads a past", "            out = m(fr)\n        else:", "            out = m(fr, {})\n        else:"),
    (T0, "the block is built before the head", "    head = JKT.build_head({})\n    if not args.get(\"no_past\"):\n        check_hook(head)\n        head.past_block = JP.PastBlock(d=JKT.arch({})[0], heads=JKT.arch({})[1])\n    return head",
     "    blk = JP.PastBlock(d=JKT.arch({})[0], heads=JKT.arch({})[1]) if not args.get(\"no_past\") else None\n    head = JKT.build_head({})\n    if blk is not None:\n        check_hook(head)\n        head.past_block = blk\n    return head"),
    (T0, "the hook is not checked", "    assert \"past\" in inspect.signature(model.forward).parameters and hasattr(model, \"past_block\"), \\\n", "    assert True or \"past\" in inspect.signature(model.forward).parameters and hasattr(model, \"past_block\"), \\\n"),
    (T0, "u is not logged", "            info[\"u\"] = float(np.mean(u_hist)) if u_hist else 0.0", "            pass"),
    (T0, "u is the last step's only", "float(np.mean(u_hist)) if u_hist else 0.0", "float(u_hist[-1]) if u_hist else 0.0"),
    (T0, "the sequence order uses the cache seed", "seed=init, args=args,", "seed=a.seed, args=args,"),
    (T0, "another recipe is not refused", "assert c.get(\"design\") == \"jpast\" and old ==", "assert c.get(\"design\") == \"jpast\" or old =="),
    (T0, "the seed twin shares the head initialisation", "    init = a.seed if a.init_seed is None else a.init_seed\n", "    init = a.seed\n"),
    ("jpast_eval.py", "the stream carries the state over a scene start", "            pos_in_scene = pos_in_scene + 1 if hp else 0\n", "            pos_in_scene = pos_in_scene + 1\n"),
    ("jpast_eval.py", "a frame without a previous keyframe reads the state", "return mode == \"off\" or not has_prev or", "return mode == \"off\" or"),
    ("jpast_eval.py", "truncated resets at the wrong frames", "(mode == \"truncated\" and pos_in_scene % window == 0)", "(mode == \"truncated\" and pos_in_scene % window == 1)"),
    ("jpast_eval.py", "off carries the state", "return mode == \"off\" or not has_prev", "return not has_prev"),
    ("jpast_eval.py", "still keeps the ego pose", "            if mode == \"still\":\n", "            if mode == \"never\":\n"),
    ("jpast_eval.py", "still moves by the inverse", "R, t = torch.eye(2, device=dev)[None].expand_as(R), torch.zeros_like(t)", "R, t = R.transpose(-1, -2), -t"),
    ("jpast_eval.py", "the key weight is the cached e at read-out", "learned = m._step_py >= JK.SWITCH", "learned = False"),
    ("jpast_eval.py", "the state is the frame before the last", "        out_prev, fr_prev = out, frd\n", "        out_prev, fr_prev = (out_prev or out), (fr_prev or frd)\n"),
    ("jpast_eval.py", "the donor is the same scene", "                t = (s + d) % len(starts)", "                t = s"),
    ("jpast_eval.py", "the donor frame is the same step", "                    donor = starts[t] + n - 1", "                    donor = starts[t] + n"),
    ("jpast_eval.py", "the donor scene is too short", "                if ends[t] - starts[t] >= n:", "                if ends[t] - starts[t] >= 0:"),
    ("jpast_eval.py", "shuffled borrows the state of this scene", "                        x, w = states[j]\n", "                        x, w = states[i]\n"),
    ("jpast_eval.py", "the stream stops late", "        if i >= last:\n            break", "        if i >= last + 1:\n            break"),
    ("jpast_eval.py", "an unkept frame is dumped", "        if i in keep:\n            frames += JE.batch_dump(out, fr)", "        if True:\n            frames += JE.batch_dump(out, fr)"),
    ("jpast_eval.py", "u by position counts the cold start", "u_by_position={f\"{lo}-{hi}\" if hi < 10 ** 9 else f\"{lo}+\": ratio((pos >= lo) & (pos <= hi) & (hp > 0))", "u_by_position={f\"{lo}-{hi}\" if hi < 10 ** 9 else f\"{lo}+\": ratio((pos >= lo) & (pos < hi))"),
    ("jpast_eval.py", "kept gained is kept lost", "g, l_ = int((ks & ~ko).sum()), int((~ks & ko).sum())", "l_, g = int((ks & ~ko).sum()), int((~ks & ko).sum())"),
    ("jpast_eval.py", "the links changed are the common links", "changed += len(link_set(fs) ^ link_set(fo))", "changed += len(link_set(fs) & link_set(fo))"),
]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def refuses(fn, text):
    """fn must fail with an AssertionError whose message holds `text` (a call that is accepted raises a RuntimeError, which no test catches)"""
    try:
        fn()
    except AssertionError as e:
        assert text in str(e), (text, str(e))
        return
    raise RuntimeError(f"accepted although it must be refused ({text})")


# ------------------------------------------------------------------------------------------------------------------------------ the stub head
def make_stub():
    import torch
    import torch.nn as nn
    import jtoken as JK

    class StubHead(nn.Module):
        """the S2 interface of `jtoken.JTokenHead` with a tiny body: forward(batch, past=None); after one projection `past_block(x, pos, dirv, valid, past)` when both exist; the output dict that `jtoken_loss` reads + x, w; set_step / _step_py / the step buffer"""

        def __init__(self, d=64):
            super().__init__()
            self.proj, self.pe, self.types, self.q, self.k, self.none, self.sc = nn.Linear(64, d), nn.Linear(4, d), nn.Linear(d, 7), nn.Linear(d, 8), nn.Linear(d, 8), nn.Linear(d, 2), nn.Linear(d, 2)
            self.register_buffer("step", torch.zeros((), dtype=torch.long))
            self._step_py = 0
            self.past_block = None

        def set_step(self, n):
            self.step.fill_(n)
            self._step_py = int(n)

        def forward(self, batch, past=None):
            valid, pos, dirv = batch["valid"].bool(), batch["pos"].float(), batch["dirv"].float()
            x = self.proj(batch["tok"].float()) + self.pe(torch.cat([pos / 50.0, dirv], -1))
            if self.past_block is not None and past is not None:
                x = self.past_block(x, pos, dirv, valid, past)
            types = self.types(x)
            g, allowed = JK.JT.pair_geometry(pos, dirv, valid, e=batch["e"].float().sigmoid())
            chord = g[..., 2] * 20.0
            cand = torch.stack([allowed & (chord <= JK.SHORT), allowed & (chord > JK.SHORT)], 1)
            z = torch.einsum("bid,bjd->bij", self.q(x), self.k(x)) / 4.0
            if self.training:
                self.step += 1
                self._step_py += 1
            return dict(logp=JK.wiring_logp(torch.stack([z, z], 1), self.none(x).permute(0, 2, 1), cand), cand=cand, scale=0.05 + self.sc(x)[:, :, None, :].expand(-1, -1, x.shape[1], -1).sigmoid(), types=types, types1=types, valid=valid,
                        chord=chord, x=x, w=1.0 - types.softmax(-1)[..., 0])
    return StubHead


def has_hook():
    import jtoken as JK
    return "past" in inspect.signature(JK.JTokenHead.forward).parameters and "past_block" in inspect.signature(JK.JTokenHead.__init__).parameters


class Patch:
    """inside the block `jtoken_train.build_head` builds the stub (None: the real head is used)"""

    def __init__(self, stub):
        import jtoken_train as JKT
        self.JKT, self.stub, self.old = JKT, stub, JKT.build_head

    def __enter__(self):
        if self.stub is not None:
            self.JKT.build_head = lambda args: self.stub()

    def __exit__(self, *a):
        self.JKT.build_head = self.old


def spy(m):
    """record every forward of the head: the step counter before it, the past and the output; `m.forward` is replaced on the instance (`restore()` puts it back)"""
    orig, calls = m.forward, []

    def forward(batch, past=None):
        c = dict(step=m._step_py, past=past, batch=batch)
        out = orig(batch) if past is None else orig(batch, past)
        calls.append(dict(c, out=out))
        return out
    m.forward = forward
    return calls, lambda: m.__dict__.pop("forward")


class NoRun:
    def log(self, *a, **k):
        pass


# ------------------------------------------------------------------------------------------------------------------------------ the trainer
def trainer_tests(T, stub, label):
    import torch
    import jpast as JP
    import jtoken as JK
    import jtoken_train as JKT
    B, S, K = 2, 3, 48
    torch.manual_seed(0)
    # the arms start from the same head: the block is built after it
    torch.manual_seed(5)
    hz = T.build_head(dict(no_past=True))
    torch.manual_seed(5)
    hr = T.build_head(dict(no_past=False))
    sz, sr = hz.state_dict(), hr.state_dict()
    assert hz.past_block is None and isinstance(hr.past_block, JP.PastBlock) and set(sr) - set(sz) == {k for k in sr if k.startswith("past_block.")} and all(torch.equal(sz[k], sr[k]) for k in sz), f"{label}: the arms do not start from the same head"
    b = T.synthetic_sequences(B, S, K, 1)
    hz.train(), hr.train()
    # arm z: the mean of the frames' E365 losses, every frame cold, the counter once per step
    hz.set_step(0)
    loss_z, infos = T.sequence_loss(hz, b, "cpu", no_past=True)
    assert hz._step_py == 1 and int(hz.step) == 1 and len(infos) == S, f"{label}: the counter of a step"
    hz.set_step(0)
    want = sum(JK.jtoken_loss(hz(b["frames"][k]), b["frames"][k])[0] for k in range(S)) / S
    assert abs(float(loss_z) - float(want)) < 1e-6, f"{label}: arm z is the mean of the frames' losses"
    # the zero-initialised block is the identity: arm r = arm z on the same weights
    hr.set_step(0)
    hz.set_step(0)
    loss_r0, _ = T.sequence_loss(hr, b, "cpu")
    loss_z2, _ = T.sequence_loss(hz, b, "cpu", no_past=True)
    assert abs(float(loss_r0) - float(loss_z2)) < 1e-6, f"{label}: the zero-initialised block changed the loss"
    # a non-zero block: the past of step k is make_past of step k - 1, step 1 has none, one counter value for the step
    blk = hr.past_block
    for p in (blk.o.weight, blk.o.bias):
        torch.nn.init.normal_(p, std=0.3)
    for s0, learned in ((0, False), (JK.SWITCH + 5, True)):
        calls, restore = spy(hr)
        hr.set_step(s0)
        loss_r, _ = T.sequence_loss(hr, b, "cpu")
        restore()
        assert len(calls) == S and calls[0]["past"] == {} and [c["step"] for c in calls] == [s0] * S and hr._step_py == s0 + 1 and int(hr.step) == s0 + 1, f"{label}: counter / first step at {s0}"
        for k in (1, 2):
            want = JP.make_past(calls[k - 1]["out"], b["frames"][k - 1], b["frames"][k], b["rel_R"][:, k], b["rel_t"][:, k], learned, b["has_prev"][:, k])
            got = calls[k]["past"]
            assert sorted(got) == sorted(want) and all(torch.equal(got[n], want[n]) for n in want) and got["x"] is calls[k - 1]["out"]["x"] and got["x"].grad_fn is not None, f"{label}: the past of step {k + 1} at counter {s0}"
        assert torch.equal(calls[1]["past"]["w"], calls[0]["out"]["w"] if learned else b["frames"][0]["e"].sigmoid()), f"{label}: the key weight at counter {s0}"
    # the gradient of the last frame's loss runs back through two steps to the first frame's states
    calls, restore = spy(hr)
    hr.set_step(0)
    T.sequence_loss(hr, b, "cpu")
    restore()
    last = JK.jtoken_loss(calls[2]["out"], b["frames"][2])[0]
    g = torch.autograd.grad(last, calls[0]["out"]["x"], allow_unused=True)[0]
    assert g is not None and g.abs().sum() > 0, f"{label}: the gradient of the last frame does not reach the first frame's state"
    # nothing of the GT enters the state: the outputs of steps 2 and 3 do not depend on the GT labels of step 1
    b2 = copy.deepcopy(b)
    f0 = b2["frames"][0]
    f0["y"], f0["ytype"], f0["ylong"], f0["yscale"] = torch.zeros_like(f0["y"]), torch.zeros_like(f0["ytype"]), torch.zeros_like(f0["ylong"]), torch.zeros_like(f0["yscale"])
    calls2, restore = spy(hr)
    hr.set_step(0)
    T.sequence_loss(hr, b2, "cpu")
    restore()
    assert all(torch.equal(calls[k]["out"]["logp"], calls2[k]["out"]["logp"]) for k in range(S)) and all(torch.equal(calls[k]["out"]["x"], calls2[k]["out"]["x"]) for k in range(S)), f"{label}: a GT label changed an output"
    # a sample without a previous keyframe inside the batch reads no state (its output is the cold output), the other sample does
    b3 = dict(b, has_prev=b["has_prev"].clone())
    b3["has_prev"][1, 1] = False
    calls3, restore = spy(hr)
    hr.set_step(0)
    T.sequence_loss(hr, b3, "cpu")
    restore()
    cold = hr(b["frames"][1], {})["logp"]
    assert torch.equal(calls3[1]["out"]["logp"][1], cold[1]) and not torch.equal(calls3[1]["out"]["logp"][0], cold[0]), f"{label}: has_prev False is not cold"
    # make_loss: u is in the info of arm r (0 at zero init), not of arm z; the mean of the last steps
    torch.manual_seed(6)
    h0 = T.build_head(dict(no_past=False))
    h0.train()
    lf = T.make_loss(NoRun(), h0, False)
    _, info = lf(h0, b)
    assert info["u"] == 0.0 and "nll_s" in info and "gpu_gb" in info, f"{label}: u at zero init"
    for p in (h0.past_block.o.weight, h0.past_block.o.bias):
        torch.nn.init.normal_(p, std=0.3)
    u1 = lf(h0, b)[1]["u"]
    u2 = lf(h0, b)[1]["u"]
    assert u1 > 0 and abs(u2 / u1 - 4.0 / 3.0) < 1e-4, f"{label}: u is the mean of the last steps: {u1} {u2} (steps 1 - 3: 0, ua, ua)"
    assert "u" not in T.make_loss(NoRun(), hz, True)(hz, b)[1], f"{label}: arm z logs u"
    # the smoke run to its end: files, the counter of the head = the steps, then the same run after a kill = the straight run
    with tempfile.TemporaryDirectory() as tmp:
        a = T.parse_args(["--name", "r", "--smoke", "--bs", "2", "--seq", "3", "--seed", "0"])
        T.train(a, runs=tmp, ckpt_every=10, smoke_k=K)
        for f in (".done", "model.pt", "config.json", "log.jsonl", "ckpt.pt"):
            assert os.path.exists(os.path.join(tmp, "r", f)), f"{label}: {f} missing"
        m, args = T.load_model("r", "cpu", tmp)
        assert m._step_py == 20 and int(m.step) == 20 and args["init_seed"] == 0 and isinstance(m.past_block, JP.PastBlock), f"{label}: the head counted {m._step_py} forwards for 20 steps"
        import json
        assert json.load(open(os.path.join(tmp, "r", "config.json")))["design"] == "jpast"
        T.train(a, runs=tmp, ckpt_every=10, smoke_k=K)                                                  # the same recipe in a finished folder is resumed (nothing left to do)
        refuses(lambda: T.train(T.parse_args(["--name", "r", "--smoke", "--bs", "2", "--seq", "3", "--seed", "1"]), runs=tmp, ckpt_every=10, smoke_k=K), "of another run")
        real_make, state = T.make_loss, dict(boom=True)

        def flaky(run, model, no_past=False, none_weight=1.0):
            f, n = real_make(run, model, no_past, none_weight), [0]

            def g(mm, bb):
                n[0] += 1
                if state["boom"] and n[0] == 13:
                    raise RuntimeError("killed")
                return f(mm, bb)
            return g
        T.make_loss = flaky
        try:
            try:
                T.train(a.__class__(**dict(vars(a), name="k")), runs=tmp, ckpt_every=10, smoke_k=K)
                raise AssertionError("the run was not killed")
            except RuntimeError as e:
                assert "killed" in str(e)
            state["boom"] = False
            T.train(a.__class__(**dict(vars(a), name="k")), runs=tmp, ckpt_every=10, smoke_k=K)
        finally:
            T.make_loss = real_make
        mk, _ = T.load_model("k", "cpu", tmp)
        assert mk._step_py == 20 and all(torch.allclose(p, q, atol=1e-5) for p, q in zip(m.state_dict().values(), mk.state_dict().values()) if p.dtype.is_floating_point), f"{label}: the resumed run is not the straight run"
        ia = T.parse_args(["--name", "i", "--smoke", "--bs", "2", "--seq", "3", "--seed", "0", "--init-seed", "3"])
        T.train(ia, runs=tmp, ckpt_every=10, smoke_k=K)
        mi, ai = T.load_model("i", "cpu", tmp)
        assert json.load(open(os.path.join(tmp, "i", "config.json")))["seed"] == 3 and ai["seed"] == 0 and ai["init_seed"] == 3 and json.load(open(os.path.join(tmp, "r", "config.json")))["seed"] == 0, f"{label}: --init-seed is the seed of the loop"
        za = T.parse_args(["--name", "z", "--smoke", "--bs", "2", "--seq", "3", "--no-past"])
        T.train(za, runs=tmp, ckpt_every=10, smoke_k=K)
        mz, _ = T.load_model("z", "cpu", tmp)
        assert mz.past_block is None and mz._step_py == 20 and not any(k.startswith("past_block") for k in mz.state_dict()), f"{label}: arm z has a block"
    # the options: arm r needs a sequence of two or more
    refuses(lambda: T.parse_args(["--name", "x", "--seq", "1"]), "at least 2")
    T.parse_args(["--name", "x", "--seq", "1", "--no-past"])


def hook_refusal(T):
    """a head without the S2 hook is refused for arm r and accepted for arm z"""
    import torch.nn as nn
    import jtoken_train as JKT

    class NoHook(nn.Module):
        def forward(self, batch):
            return {}
    old = JKT.build_head
    JKT.build_head = lambda args: NoHook()
    try:
        refuses(lambda: T.build_head(dict(no_past=False)), "S2 hook")
        assert isinstance(T.build_head(dict(no_past=True)), NoHook)
    finally:
        JKT.build_head = old


# ------------------------------------------------------------------------------------------------------------------------------ the read-out
def eval_tests(E, T, stub, label):
    import numpy as np
    import torch
    import jpast as JP
    import jtoken as JK
    K = 48
    torch.manual_seed(1)
    m = T.build_head(dict(no_past=False))
    for p in (m.past_block.o.weight, m.past_block.o.bias):
        torch.nn.init.normal_(p, std=0.3)
    m.set_step(JK.SWITCH + 10)
    m.eval()
    hp = [False, True, True, True, False, True, True, True, True]
    prev = [-1, 0, 1, 2, -1, 4, 5, 6, 7]
    n = len(hp)
    g = torch.Generator().manual_seed(3)
    th = torch.randn(n, generator=g) * 0.3
    R = torch.stack([torch.stack([th.cos(), -th.sin()], -1), torch.stack([th.sin(), th.cos()], -1)], -2)
    t = torch.randn(n, 2, generator=g) * 3.0
    frames = [dict(JK.with_tokens_np(JK.synthetic_batch(1, K, seed=10 + i)), token=[f"t{i}"], row=torch.tensor([i])) for i in range(n)]
    batches = lambda: [dict(frames=[frames[i]], rel_R=R[i][None, None], rel_t=t[i][None, None], has_prev=torch.tensor([[hp[i]]]), idx=torch.tensor([[i]])) for i in range(n)]  # noqa: E731

    def manual(mode, window=3):
        """the read-out by hand: the outputs per frame"""
        outs, prev_out, pos = [], None, 0
        for i in range(n):
            pos = pos + 1 if hp[i] else 0
            cold = mode == "off" or not hp[i] or (mode == "truncated" and pos % window == 0)
            if cold:
                past = {}
            else:
                Ri, ti = (torch.eye(2)[None], torch.zeros(1, 2)) if mode == "still" else (R[i][None], t[i][None])
                past = JP.make_past(prev_out, frames[i - 1], frames[i], Ri, ti, True)
            with torch.no_grad():
                prev_out = m(frames[i], past)
            outs.append(prev_out)
        return outs
    for mode in ("stream", "off", "truncated", "still"):
        want = manual(mode)
        got = {i: out for i, _, out, _ in E.iter_outputs(m, batches(), mode, 3)}
        assert all(torch.equal(got[i]["logp"], want[i]["logp"]) for i in range(n)), f"{label}: mode {mode} is not the hand loop"
    cold = [m(frames[i], {})["logp"] for i in range(n)]
    stream = {i: (out, info) for i, _, out, info in E.iter_outputs(m, batches(), "stream", 3)}
    assert all(torch.equal(stream[i][0]["logp"], cold[i]) for i in (0, 4)) and not any(torch.equal(stream[i][0]["logp"], cold[i]) for i in (1, 2, 3, 5, 6, 7, 8)), f"{label}: cold starts / the past changes the output"
    assert [stream[i][1]["pos"] for i in range(n)] == [0, 1, 2, 3, 0, 1, 2, 3, 4] and [stream[i][1]["has_prev"] for i in range(n)] == hp and abs(stream[1][1]["shift"] - float(t[1].norm())) < 1e-6
    assert all(stream[i][1]["n"] == 0 and stream[i][1]["d2"] == 0.0 for i in (0, 4)) and all(stream[i][1]["n"] > 0 and stream[i][1]["d2"] > 0 and stream[i][1]["x2"] > 0 for i in (1, 2, 3, 5)), f"{label}: u sums"
    trunc = {i: out for i, _, out, _ in E.iter_outputs(m, batches(), "truncated", 3)}
    assert torch.equal(trunc[3]["logp"], cold[3]) and torch.equal(trunc[7]["logp"], cold[7]), f"{label}: truncated is cold at position 3"
    assert all(torch.equal(trunc[i]["logp"], stream[i][0]["logp"]) for i in (1, 2, 5, 6)) and not any(torch.equal(trunc[i]["logp"], stream[i][0]["logp"]) for i in (3, 7, 8)), f"{label}: truncated = stream until the reset, then another state"
    # shuffled: the donor plan by hand, the state of the donor frame (from a stream pass) moved by this frame's pose
    plan = E.donor_plan(prev)
    assert plan == [None, 4, 5, 6, None, 0, 1, 2, 3], plan
    assert E.donor_plan([-1, 0, 1, 2, -1, 4]) == [None, 4, 5, None, None, 0], E.donor_plan([-1, 0, 1, 2, -1, 4])      # the donor scene (2 frames) has no step 3
    assert E.donor_plan([-1, 0, 1]) == [None, None, None], "a single scene has no donor"
    states = {i: (stream[i][0]["x"], stream[i][0]["w"]) for i in range(n)}
    sh = {i: out for i, _, out, _ in E.iter_outputs(m, batches(), "shuffled", 3, donor=plan, states=states, frame_of=lambda j: frames[j])}
    for i in (1, 5):
        j = plan[i]
        past = JP.make_past(dict(x=states[j][0], w=states[j][1]), frames[j], frames[i], R[i][None], t[i][None], True)
        with torch.no_grad():
            assert torch.equal(sh[i]["logp"], m(frames[i], past)["logp"]), f"{label}: shuffled frame {i}"
    assert torch.equal(sh[0]["logp"], cold[0]) and torch.equal(sh[4]["logp"], cold[4]) and not torch.equal(sh[1]["logp"], stream[1][0]["logp"])
    # dump_frames: only the asked frames, in order, the stream carried through the others and stopped after the last one; the use arrays aligned
    seen = []

    def counting():
        for bb in batches():
            seen.append(int(bb["idx"][0, 0]))
            yield bb
    fr_d, use = E.dump_frames(m, counting(), {0, 2, 3, 5}, "stream", 3, "cpu", log=lambda *a, **k: None)
    assert [f["token"] for f in fr_d] == ["t0", "t2", "t3", "t5"] and use["idx"] == [0, 2, 3, 5] and use["pos"] == [0, 2, 3, 1] and use["has_prev"] == [False, True, True, True] and seen == [0, 1, 2, 3, 4, 5], f"{label}: dump_frames {use} {seen}"
    assert use["d2"][0] == 0.0 and use["d2"][1] > 0 and all(len(v) == 4 for v in use.values())
    assert torch.equal(torch.from_numpy(np.asarray(fr_d[1]["pnone"])), torch.from_numpy(np.asarray(E.JE.batch_dump(stream[2][0], frames[2])[0]["pnone"]))), f"{label}: the dumped frame is the stream output"
    # the writer of jtoken_eval accepts a dumped frame (the nodes are the kept tokens', no GT field)
    pred = E.JE.frame_prediction(frames[2]["tokens_np"][0], fr_d[1], "rec")
    assert pred["assign"] is None and pred["maps"] is None
    # arm z: no block, every frame the plain head on the frame
    mz = T.build_head(dict(no_past=True))
    mz.set_step(JK.SWITCH + 10)
    mz.eval()
    for i, _, out, info in E.iter_outputs(mz, batches(), "stream", 3):
        with torch.no_grad():
            assert torch.equal(out["logp"], mz(frames[i])["logp"]) and info["n"] == 0, f"{label}: arm z frame {i}"
    # the use report by hand
    mk = lambda keeps, links, d2, x2, pos, hp_: dict(meta=dict(run="r", split="val", use=dict(d2=d2, x2=x2, pos=pos, has_prev=hp_)), frames=[dict(token=f"t{i}", keep=np.array(k), rec=dict(src=np.array([a for a, _ in l_], int), dst=np.array([b for _, b in l_], int))) for i, (k, l_) in enumerate(zip(keeps, links))])  # noqa: E731
    s_ = mk([[1, 1, 0], [1, 0, 0]], [[(0, 1), (1, 2)], [(0, 1)]], [0.0, 1.0], [4.0, 4.0], [0, 1], [False, True])
    o_ = mk([[1, 0, 1], [1, 0, 0]], [[(0, 1)], [(0, 1)]], [0.0, 0.0], [4.0, 4.0], [0, 0], [False, False])
    rep = E.use_report(s_, o_)
    assert abs(rep["u"] - 0.5) < 1e-9 and rep["frames_with_past"] == 1 and abs(rep["u_by_position"]["1-1"] - 0.5) < 1e-9 and np.isnan(rep["u_by_position"]["2-2"]) and rep["against_off"] == dict(kept_gained_per_frame=0.5, kept_lost_per_frame=0.5, kept_flips_per_frame=1.0,
                                                                                                                                                                 frames_with_a_flip=0.5, links_changed_per_frame=0.5), rep
    assert E.is_cold("stream", True, 5, 3) is False and E.is_cold("stream", False, 0, 3) and E.is_cold("off", True, 5, 3) and E.is_cold("truncated", True, 6, 3) and not E.is_cold("truncated", True, 5, 3) and not E.is_cold("still", True, 3, 3)


def dump_on_real_frames(E, T, stub):
    """the dump command on 12 real VAL frames of the cache with an untrained head: names, meta, the writer of jtoken_eval, the modes differ"""
    import pickle
    import numpy as np
    import torch
    import jtoken as JK
    with tempfile.TemporaryDirectory() as tmp:
        torch.manual_seed(2)
        m = T.build_head(dict(no_past=False))
        for p in (m.past_block.o.weight, m.past_block.o.bias):
            torch.nn.init.normal_(p, std=0.3)
        m.set_step(JK.SWITCH)
        os.makedirs(os.path.join(tmp, "x"))
        args = dict(seed=0, ident="e292_a", K=None, cache=None, steps=8000, no_past=False, init_seed=0)
        torch.save(dict(state=m.state_dict(), args=args, steps=8000), os.path.join(tmp, "x", "model.pt"))
        paths = {}
        for mode in ("stream", "off", "truncated"):
            paths[mode] = E.dump(argparse.Namespace(run="x", seed=0, split="val", mode=mode, window=3, cpu=True, max=12), runs=tmp)
        assert [os.path.basename(p) for p in paths.values()] == ["token_val_n12.pkl", "token_val_off_n12.pkl", "token_val_truncated_n12.pkl"]
        refuses(lambda: E.dump(argparse.Namespace(run="x", seed=0, split="val", mode="stream", window=3, cpu=True, max=12), runs=tmp), "never overwritten")
        d = {k: pickle.load(open(p, "rb")) for k, p in paths.items()}
        meta = d["stream"]["meta"]
        assert {"run", "seed", "split", "drop", "drop_seed", "idx", "every", "tokens", "K", "ident", "cache", "steps"} <= set(meta) and meta["idx"] == list(range(12)) and meta["mode"] == "stream" and meta["every"] == 1 and meta["drop"] == 0.0
        assert [f["token"] for f in d["stream"]["frames"]] == meta["tokens"] and len(d["stream"]["frames"]) == 12 and meta["use"]["has_prev"][0] is False
        import lane_ptok_data as LP
        ds = LP.PTokData("val", 0, "e292_a")
        pred = E.JE.frame_prediction(ds[3]["tokens_np"], d["stream"]["frames"][3], "rec")
        assert pred["assign"] is None and pred["maps"] is None
        assert all(np.array_equal(d["stream"]["frames"][0][k], d["off"]["frames"][0][k]) for k in ("pnone", "keep")), "frame 0 has no past: stream = off"
        assert not np.array_equal(d["stream"]["frames"][3]["pnone"], d["off"]["frames"][3]["pnone"]), "the past changed nothing at frame 3"
        assert np.array_equal(d["truncated"]["frames"][2]["pnone"], d["stream"]["frames"][2]["pnone"]) and not np.array_equal(d["truncated"]["frames"][3]["pnone"], d["stream"]["frames"][3]["pnone"]), "truncated: window 3"
        rep = E.use_report(d["stream"], d["off"])
        assert rep["frames_with_past"] == 11 and rep["u"] > 0 and rep["against_off"]["kept_flips_per_frame"] >= 0


def selftest(train_path=TRAIN, eval_path=EVAL):
    import jtoken_train as JKT
    assert JKT.build_head is not None
    T, E = load(train_path, "jpast_train_under_test"), load(eval_path, "jpast_eval_under_test")
    hook_refusal(T)
    heads = [("stub", make_stub())] + ([("real", None)] if has_hook() else [])
    for label, stub in heads:
        with Patch(stub):
            trainer_tests(T, stub, label)
            eval_tests(E, T, stub, label)
        print(f"jpast trainer + read-out tests: the {label} head ok", flush=True)
    with Patch(make_stub()):
        dump_on_real_frames(E, T, make_stub())
    print("jpast_train selftest ok" + ("" if has_hook() else " (the real head has no S2 hook yet: the stub only)"))


# ------------------------------------------------------------------------------------------------------------------------------ mutants
def one(args):
    k, (mod, name, a, b), tmp = args
    path = TRAIN if mod == T0 else EVAL
    src = open(path, encoding="utf-8").read()
    if src.count(a) != 1:
        return name, f"NOT APPLIED ({src.count(a)} matches)"
    d = os.path.join(tmp, str(k))
    os.makedirs(d)
    p = os.path.join(d, mod)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(src.replace(a, b))
    other = [TRAIN, EVAL]
    other[mod != T0] = p
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "one", other[0], other[1]], capture_output=True, text=True, env=ENV)
    return name, "killed" if r.returncode != 0 else "SURVIVED"


def run_mutants(workers=4):
    left = []
    with tempfile.TemporaryDirectory() as tmp, cf.ThreadPoolExecutor(workers) as ex:
        for name, res in ex.map(one, [(k, m, tmp) for k, m in enumerate(MUTANTS)]):
            print(f"{res:<12} {name}", flush=True)
            if res != "killed":
                left.append(name)
    print(f"{len(MUTANTS) - len(left)} of {len(MUTANTS)} killed", flush=True)
    return left


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        selftest(sys.argv[2], sys.argv[3])
        sys.exit(0)
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "one", TRAIN, EVAL], capture_output=True, text=True, env=ENV)
    assert r.returncode == 0, f"the selftest failed:\n{r.stdout[-1500:]}{r.stderr[-2500:]}"
    print(r.stdout.strip())
    if len(sys.argv) < 2 or sys.argv[1] != "selftests":
        left = run_mutants()
        assert not left, f"mutants survived: {left}"
    print("test_jpast_train ok")
