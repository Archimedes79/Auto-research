"""Tests of the E368 trainer (`train/jpast_train.py`) and the read-out (`evaluation/jpast_eval.py`).

    wsl bash tools/env/_py14.sh test_jpast_train.py            # the selftest (a stub head with the S2 interface, and the real `jtoken.JTokenHead` when it has the hook), then the hand mutants of both files: every mutant must make it FAIL
    wsl bash tools/env/_py14.sh test_jpast_train.py selftests  # the selftest and one batch of the real TRAIN chains only

The loop by hand: the head's step counter held for the forwards of a step and advanced once, the past of step k = `jpast.make_past` of the frames and the poses of step k - 1 (the key weight before and after the switch), the state in the graph (the gradient of the last frame reaches
the first), no GT key in the state, a sample without a previous keyframe cold, arm z = the mean of the frames' losses, the zero-initialised block = arm z, `u` logged, the smoke run with its resume, the hook check. The read-out: the modes stream / truncated / off / still / shuffled against a hand loop,
the donor plan, the cold starts, the dump of the cache frames of VAL (a toy untrained head on 12 real frames: the writer of `jtoken_eval` accepts it). A mutant = one textual change of the source of `jpast_train.py` or `jpast_eval.py`; the mutated copy is loaded in place of the file, four at a time.
"""
import argparse
import concurrent.futures as cf
import copy
import importlib.util
import inspect
import math
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
    (T0, "the forwards of a step see different counters", "        m.set_step(s0)\n        if no_past:", "        if no_past:"),
    (T0, "the pose of the wrong step", "frames[k - 1], fr, R[:, k], t[:, k], learned, hp[:, k])", "frames[k - 1], fr, R[:, k - 1], t[:, k - 1], learned, hp[:, k])"),
    (T0, "the pose is the inverse (the frames swapped)", "JP.make_past(out_prev, frames[k - 1], fr, R[:, k], t[:, k], learned, hp[:, k])", "JP.make_past(out_prev, fr, frames[k - 1], R[:, k], t[:, k], learned, hp[:, k])"),
    (T0, "the state of the previous step is detached", "        out_prev = out\n", "        out_prev = dict(out, x=out['x'].detach(), w=out['w'].detach())\n"),
    (T0, "the first step reads a past", "past = {} if k == 0 else", "past = {} if k == 99 else"),
    (T0, "has_prev is ignored", "learned, hp[:, k])", "learned)"),
    (T0, "the loss is the sum of the frames", "return sum(losses) / len(losses), infos", "return sum(losses), infos"),
    (T0, "the loss is that of the last frame only", "return sum(losses) / len(losses), infos", "return losses[-1], infos"),
    (T0, "the block is built before the head", "    head = JKT.build_head({})\n    if not args.get(\"no_past\"):\n        check_hook(head)\n        head.past_block = BLOCKS[args.get(\"block\") or \"residual\"](d=JKT.arch({})[0], heads=JKT.arch({})[1])\n    return head",
     "    blk = BLOCKS[args.get(\"block\") or \"residual\"](d=JKT.arch({})[0], heads=JKT.arch({})[1]) if not args.get(\"no_past\") else None\n    head = JKT.build_head({})\n    if blk is not None:\n        check_hook(head)\n        head.past_block = blk\n    return head"),
    (T0, "the hook is not checked", "    assert \"past\" in inspect.signature(model.forward).parameters and hasattr(model, \"past_block\"), \\\n", "    assert True or \"past\" in inspect.signature(model.forward).parameters and hasattr(model, \"past_block\"), \\\n"),
    (T0, "u is not logged", "            info[\"u\"], info[\"u_none\"] = (float(np.mean(h)) if h else 0.0 for h in (u_hist, un_hist))", "            pass"),
    (T0, "u is the last step's only", "(float(np.mean(h)) if h else 0.0 for h in (u_hist, un_hist))", "(float(h[-1]) if h else 0.0 for h in (u_hist, un_hist))"),
    (T0, "the sequence order uses the cache seed", "seed=init, args=args,", "seed=a.seed, args=args,"),
    (T0, "another recipe is not refused", "assert c.get(\"design\") == \"jpast\" and old ==", "assert c.get(\"design\") == \"jpast\" or old =="),
    (T0, "the seed twin shares the head initialisation", "    init = a.seed if a.init_seed is None else a.init_seed\n", "    init = a.seed\n"),
    (T0, "the switch is judged one step late", "learned = s0 >= JK.SWITCH", "learned = s0 > JK.SWITCH"),
    (T0, "U_WINDOW is 10", "U_WINDOW = 100 ", "U_WINDOW = 10 "),
    (T0, "a change of seq is not refused at a resume", 'RECIPE = ("seed", "init_seed", "cache", "ident", "K", "bs", "seq", "steps", "lr", "no_past", "block")', 'RECIPE = ("seed", "init_seed", "cache", "ident", "K", "bs", "steps", "lr", "no_past", "block")'),
    (T0, "the block is not part of the recipe", 'RECIPE = ("seed", "init_seed", "cache", "ident", "K", "bs", "seq", "steps", "lr", "no_past", "block")', 'RECIPE = ("seed", "init_seed", "cache", "ident", "K", "bs", "seq", "steps", "lr", "no_past")'),
    (T0, "the fused option builds the residual block", 'BLOCKS = {"residual": JP.PastBlock, "fused": JP.FusedPastBlock}', 'BLOCKS = {"residual": JP.PastBlock, "fused": JP.PastBlock}'),
    (T0, "a config without a block is not the residual block", 'r["block"] = r["block"] or "residual"', 'r["block"] = r["block"]'),
    (T0, "arm z accepts a block", 'assert not (args["no_past"] and (args.get("block") or "residual") == "fused"), ', 'assert True or '),
    (T0, "the loader gets K_seq 3", "K_seq=a.seq, ident=a.ident", "K_seq=3, ident=a.ident", "real"),
    (T0, "the loader gets the head seed", 'LPD.loader("train", a.seed, a.bs', 'LPD.loader("train", init, a.bs', "real"),
    (T0, "the loader gets another batch size", "a.seed, a.bs, a.workers", "a.seed, 4, a.workers", "real"),
    ("jpast_eval.py", "the read-out's switch is one step late", "learned = m._step_py >= JK.SWITCH", "learned = m._step_py > JK.SWITCH"),
    ("jpast_eval.py", "the last scene ends one frame early", "ends = starts[1:] + [len(prev)]", "ends = starts[1:] + [len(prev) - 1]"),
    ("jpast_eval.py", "dump does not check the seed", 'assert args["seed"] == a.seed,', 'assert True or args["seed"] == a.seed,'),
    ("jpast_eval.py", "the sibling folder has no model.pt", "shutil.copyfile(src, dst)", "pass"),
    ("jpast_eval.py", "a sibling with another model.pt is accepted", 'assert filecmp.cmp(src, dst, shallow=False), f"{dst} is not a copy of {src}"', "pass"),
    ("jpast_eval.py", "the meta names the run, not the sibling", "meta = dict(run=name,", "meta = dict(run=a.run,"),
    ("jpast_eval.py", "every mode goes to a sibling", 'return run if mode == "stream" else f"{run}_{mode}"', 'return f"{run}_{mode}"'),
    ("jpast_eval.py", "use reads the off dump of the run itself", "dump_file(runs, run_folder(a.run, m), a.split, a.max)", "dump_file(runs, a.run, a.split, a.max)"),
    ("jpast_eval.py", "the state count forgets the channels", 'state_n=int(v.sum()) * int(out["x"].shape[-1])', "state_n=int(v.sum())"),
    ("jpast_eval.py", "the carried state is the block input", 'state_ss=float((out["x"][0].float() ** 2)[v].sum())', "state_ss=x2"),
    ("jpast_eval.py", "the first 5 frames are 4", "FIRST = 5 ", "FIRST = 4 "),
    ("jpast_eval.py", "time: a frame with a past is timed as cold", "(row_warm if (past or block is None) else row_cold)", "(row_cold if (past or block is None) else row_warm)"),
    ("jpast_eval.py", "time: the untimed pass is timed", "        if rep:\n            warm += row_warm", "        if True:\n            warm += row_warm"),
    ("jpast_eval.py", "time: the state of the previous frame is not carried", "                past = JP.make_past(out_prev, fr_prev, frd, R_, t_, m._step_py >= JK.SWITCH)", "                past = {}"),
    ("jpast_eval.py", "the state RMS stops one position early", "for p in range(int(pos.max()) + 1)]", "for p in range(int(pos.max()))]"),
    (T0, "arm r with one sequence per step is accepted", 'assert args["no_past"] or args["bs"] >= 2,', 'assert True or args["bs"] >= 2,'),
    (T0, "u is logged as u_none", "(float(np.mean(h)) if h else 0.0 for h in (u_hist, un_hist))", "(float(np.mean(h)) if h else 0.0 for h in (u_hist, u_hist))"),
    (T0, "u takes the none-key value", 'u_hist.append(u["u"])', 'u_hist.append(u["u_none"])'),
    (T0, "the window of u_none is 10", "collections.deque(maxlen=U_WINDOW), collections.deque(maxlen=U_WINDOW)", "collections.deque(maxlen=U_WINDOW), collections.deque(maxlen=10)"),
    ("jpast_eval.py", "the meter gets no other scene's past", "meter.donor = other", "meter.donor = None"),
    ("jpast_eval.py", "the meter compares with the stream's own past", "meter.donor = other", "meter.donor = past"),
    ("jpast_eval.py", "mode shuffled ignores its donor", "past = other or {}", "past = {}"),
    ("jpast_eval.py", "the stream dump makes no pass for the other scenes' states", 'if a.split == "val" and a.mode != "off" and not args.get("no_past"):', 'if a.mode == "shuffled":'),
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
    ("jpast_eval.py", "shuffled borrows the state of this scene", "dict(x=states[j][0].to(dev), w=states[j][1].to(dev))", "dict(x=states[i][0].to(dev), w=states[i][1].to(dev))"),
    ("jpast_eval.py", "the stream stops late", "        if i >= last:\n            break", "        if i >= last + 1:\n            break"),
    ("jpast_eval.py", "an unkept frame is dumped", "        if i in keep:\n            frames += JE.batch_dump(out, fr)", "        if True:\n            frames += JE.batch_dump(out, fr)"),
    ("jpast_eval.py", "the u buckets exclude their upper bound", "bucket = {f\"{lo}-{hi}\" if hi < 10 ** 9 else f\"{lo}+\": (pos >= lo) & (pos <= hi) & (hp > 0)", "bucket = {f\"{lo}-{hi}\" if hi < 10 ** 9 else f\"{lo}+\": (pos >= lo) & (pos < hi) & (hp > 0)"),
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

        def _load_from_state_dict(self, *a, **k):
            super()._load_from_state_dict(*a, **k)
            self._step_py = int(self.step)

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
    torch.manual_seed(5)
    hf = T.build_head(dict(no_past=False, block="fused"))
    sz, sr = hz.state_dict(), hr.state_dict()
    assert isinstance(hf.past_block, JP.FusedPastBlock) and isinstance(T.build_head(dict(no_past=False, block=None)).past_block, JP.PastBlock) and isinstance(hr.past_block, JP.PastBlock), f"{label}: --block picks the block"
    assert all(torch.equal(sz[k], hf.state_dict()[k]) for k in sz), f"{label}: the fused arm does not start from the same head"
    refuses(lambda: T.check_options(dict(bs=4, steps=1, lr=1e-3, seq=3, no_past=True, block="fused")), "contradicts")
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
    for s0, learned in ((0, False), (JK.SWITCH - 1, False), (JK.SWITCH, True), (JK.SWITCH + 5, True)):                # the switch at the boundary: step 999 is not learned, 1000 is
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
    if stub is None:                                                                                        # the real head switches its own key weights at the same boundary as `make_past` (the stub has no switch)
        hb = T.build_head(dict(no_past=True))
        hb.eval()
        lp = {}
        for s0 in (0, JK.SWITCH - 1, JK.SWITCH):
            hb.set_step(s0)
            lp[s0] = hb(b["frames"][0])["logp"]
        assert torch.equal(lp[0], lp[JK.SWITCH - 1]) and not torch.equal(lp[0], lp[JK.SWITCH]), f"{label}: the head's own switch is not at step {JK.SWITCH}"
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
    assert info["u"] == 0.0 and info["u_none"] == 0.0 and "nll_s" in info and "gpu_gb" in info, f"{label}: u at zero init"
    for p in (h0.past_block.o.weight, h0.past_block.o.bias):
        torch.nn.init.normal_(p, std=0.3)
    i1, i2 = lf(h0, b)[1], lf(h0, b)[1]
    assert i1["u"] > 0 and i1["u_none"] > 0 and abs(i2["u"] / i1["u"] - 4.0 / 3.0) < 1e-4 and abs(i2["u_none"] / i1["u_none"] - 4.0 / 3.0) < 1e-4, f"{label}: u and u_none are the means of the last steps: {i1} {i2} (steps 1 - 3: 0, ua, ua)"
    assert i1["u"] != i1["u_none"], f"{label}: u is not u_none"
    iz = T.make_loss(NoRun(), hz, True)(hz, b)[1]
    assert "u" not in iz and "u_none" not in iz, f"{label}: arm z logs u"
    # u is the mean of the last U_WINDOW = 100 steps = the card's last 500 steps as the five 100-step rows of log.jsonl (`trainer.fit` logs every 100)
    assert T.U_WINDOW == 100 == inspect.signature(T.T.fit).parameters["log_every"].default and 5 * T.U_WINDOW == 500, f"{label}: U_WINDOW {T.U_WINDOW}"
    if stub is not None:
        import numpy as np
        h1 = T.build_head(dict(no_past=False))
        h1.train()
        for p in (h1.past_block.o.weight, h1.past_block.o.bias):
            torch.nn.init.normal_(p, std=0.3)
        rec, orig = [], T.JP.UseMeter.take

        def take(self):
            r = orig(self)
            rec.append((r["u"], r["u_none"]) if r else None)
            return r
        T.JP.UseMeter.take = take
        try:
            lf1 = T.make_loss(NoRun(), h1, False)
            for _ in range(120):
                with torch.no_grad():
                    h1.past_block.o.weight.mul_(1.02)                                                       # u grows from step to step: the window can be seen
                info1 = lf1(h1, b)[1]
        finally:
            T.JP.UseMeter.take = orig
        for j_, key in ((0, "u"), (1, "u_none")):
            vals = [r_[j_] for r_ in rec]
            assert abs(info1[key] - float(np.mean(vals[-100:]))) < 1e-9 and abs(info1[key] - float(np.mean(vals[-10:]))) > 1e-3, f"{label}: {key} is not the mean of the last 100 steps"
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
        cfg, folder = json.load(open(os.path.join(tmp, "r", "config.json")))["args"], os.path.join(tmp, "r")
        change = dict(seed=lambda v: v + 1, init_seed=lambda v: v + 1, cache=lambda v: "another_root", ident=lambda v: "e273_p", K=lambda v: 384, bs=lambda v: v + 1, seq=lambda v: v + 1, steps=lambda v: v + 1, lr=lambda v: v * 2, no_past=lambda v: not v,
                      block=lambda v: "fused" if v != "fused" else "residual")
        assert set(change) == set(T.RECIPE), f"{label}: a recipe key without a refusal test: {set(T.RECIPE) ^ set(change)}"
        T.check_start(cfg, folder)                                                                       # the same recipe is accepted
        old_cfg = json.load(open(os.path.join(folder, "config.json")))                                   # E368's configs have no block key = the residual block
        del old_cfg["args"]["block"]
        os.makedirs(os.path.join(tmp, "old"))
        json.dump(old_cfg, open(os.path.join(tmp, "old", "config.json"), "w"))
        T.check_start(dict(cfg, block="residual"), os.path.join(tmp, "old"))
        refuses(lambda: T.check_start(dict(cfg, block="fused"), os.path.join(tmp, "old")), "of another run")
        af = T.parse_args(["--name", "f", "--smoke", "--bs", "2", "--seq", "3", "--seed", "0", "--block", "fused"])      # the fused arm trains 20 steps on the smoke batches
        T.train(af, runs=tmp, ckpt_every=10, smoke_k=K)
        mf, argsf = T.load_model("f", "cpu", tmp)
        assert isinstance(mf.past_block, JP.FusedPastBlock) and argsf["block"] == "fused" and mf._step_py == 20, f"{label}: the fused smoke run"
        for k_ in change:                                                                                # a change of EVERY recipe key is refused at a resume
            refuses(lambda: T.check_start(dict(cfg, **{k_: change[k_](cfg[k_])}), folder), "of another run")
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
    refuses(lambda: T.parse_args(["--name", "x", "--bs", "1"]), "at least 2 sequences")                    # u compares with the past of another sample of the batch
    T.parse_args(["--name", "x", "--seq", "1", "--no-past"])
    T.parse_args(["--name", "x", "--bs", "1", "--no-past"])


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
    warm, cold = E.time_head(m, batches(), "cpu", 2)                                                     # the timing loop: warm = frames with a past, cold = the scene starts, the first pass is not counted
    assert len(warm) == 2 * sum(hp) and len(cold) == 2 * (n - sum(hp)) and min(warm + cold) > 0, f"{label}: time_head counts {len(warm)} {len(cold)}"
    mz0 = T.build_head(dict(no_past=True)).eval()
    warm0, cold0 = E.time_head(mz0, batches(), "cpu", 1)
    assert len(warm0) == n and cold0 == [], f"{label}: a head without a block is all warm"

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
    assert all(stream[i][1]["n"] == 0 and stream[i][1]["d2_none"] == 0.0 for i in (0, 4)) and all(stream[i][1]["n"] > 0 and stream[i][1]["d2_none"] > 0 and stream[i][1]["x2"] > 0 for i in (1, 2, 3, 5)), f"{label}: u_none sums"
    assert all(stream[i][1]["n_other"] == 0 and stream[i][1]["d2_other"] == 0.0 for i in range(n)), f"{label}: a batch of one sample without a donor has no u"
    for i in (0, 2):                                                                                         # the CARRIED state: the final token states x of the frame's valid tokens (not the block's input)
        o_i, inf = stream[i]
        v_i = o_i["valid"][0].bool()
        assert inf["state_n"] == int(v_i.sum()) * 64 and inf["state_ss"] > 0 and abs(inf["state_ss"] - float((o_i["x"][0] ** 2)[v_i].sum())) < 1e-4 * inf["state_ss"] and abs(inf["state_ss"] - inf["x2"]) > 1e-3 * inf["state_ss"], f"{label}: the carried state's sums"
    for s0, learned in ((JK.SWITCH - 1, False), (JK.SWITCH, True)):                                          # the read-out's switch of the key weight at the boundary (as the head's and make_past's)
        m.set_step(s0)
        calls_b, restore = spy(m)
        list(E.iter_outputs(m, batches()[:2], "stream", 3))
        restore()
        assert torch.equal(calls_b[1]["past"]["w"], calls_b[0]["out"]["w"] if learned else frames[0]["e"].sigmoid()), f"{label}: the read-out's key weight at counter {s0}"
    m.set_step(JK.SWITCH + 10)
    trunc = {i: out for i, _, out, _ in E.iter_outputs(m, batches(), "truncated", 3)}
    assert torch.equal(trunc[3]["logp"], cold[3]) and torch.equal(trunc[7]["logp"], cold[7]), f"{label}: truncated is cold at position 3"
    assert all(torch.equal(trunc[i]["logp"], stream[i][0]["logp"]) for i in (1, 2, 5, 6)) and not any(torch.equal(trunc[i]["logp"], stream[i][0]["logp"]) for i in (3, 7, 8)), f"{label}: truncated = stream until the reset, then another state"
    # shuffled: the donor plan by hand, the state of the donor frame (from a stream pass) moved by this frame's pose
    plan = E.donor_plan(prev)
    assert plan == [None, 4, 5, 6, None, 0, 1, 2, 3], plan
    assert E.donor_plan([-1, 0, 1, 2, -1, 4]) == [None, 4, 5, None, None, 0], E.donor_plan([-1, 0, 1, 2, -1, 4])      # the donor scene (2 frames) has no step 3
    assert E.donor_plan([-1, 0, 1]) == [None, None, None], "a single scene has no donor"
    states = {i: (stream[i][0]["x"], stream[i][0]["w"]) for i in range(n)}
    shi = {i: (out, info) for i, _, out, info in E.iter_outputs(m, batches(), "shuffled", 3, donor=plan, states=states, frame_of=lambda j: frames[j])}
    sh = {i: o_[0] for i, o_ in shi.items()}
    assert all(shi[i][1]["n_other"] == 0 for i in range(n)) and all(shi[i][1]["n"] > 0 for i in (1, 5)), f"{label}: in mode shuffled the donor is the past, no other-scene comparison"
    for i in (1, 5):
        j = plan[i]
        past = JP.make_past(dict(x=states[j][0], w=states[j][1]), frames[j], frames[i], R[i][None], t[i][None], True)
        with torch.no_grad():
            assert torch.equal(sh[i]["logp"], m(frames[i], past)["logp"]), f"{label}: shuffled frame {i}"
    assert torch.equal(sh[0]["logp"], cold[0]) and torch.equal(sh[4]["logp"], cold[4]) and not torch.equal(sh[1]["logp"], stream[1][0]["logp"])
    # u in the stream read-out: the update with the stream's past against the update with the OTHER scene's past (the donor of the plan, warped by this frame's pose), by hand from the block's own inputs
    cap = {}
    hk = m.past_block.register_forward_hook(lambda mod, args, out: cap.__setitem__("args", args))
    try:
        for i, _, out, info in E.iter_outputs(m, batches(), "stream", 3, donor=plan, states=states, frame_of=lambda j: frames[j]):
            if not hp[i]:
                assert info["n_other"] == 0 and info["d2_other"] == 0.0 and info["x2_other"] == 0.0, f"{label}: frame {i} has no past"
                continue
            xin, pi, di, vi, pa = cap["args"]
            j = plan[i]
            pd_ = JP.make_past(dict(x=states[j][0], w=states[j][1]), frames[j], frames[i], R[i][None], t[i][None], True)
            with torch.no_grad():
                y_s, y_o, y_n = (m.past_block.delta(xin, pi, di, vi, q_) for q_ in (pa, pd_, None))
            mv = vi[..., None].float()
            want_o, want_n = float((((y_s - y_o) * mv) ** 2).sum()), float((((y_s - y_n) * mv) ** 2).sum())
            assert info["n_other"] == int(vi.sum()) == info["n"] and abs(info["d2_other"] - want_o) < 1e-4 * max(want_o, 1) and abs(info["d2_none"] - want_n) < 1e-4 * max(want_n, 1) and want_o > 0 and info["x2_other"] == info["x2"], f"{label}: u of frame {i}: {info} {want_o} {want_n}"
    finally:
        hk.remove()
    # dump_frames: only the asked frames, in order, the stream carried through the others and stopped after the last one; the use arrays aligned
    seen = []

    def counting():
        for bb in batches():
            seen.append(int(bb["idx"][0, 0]))
            yield bb
    fr_d, use = E.dump_frames(m, counting(), {0, 2, 3, 5}, "stream", 3, "cpu", log=lambda *a, **k: None)
    assert [f["token"] for f in fr_d] == ["t0", "t2", "t3", "t5"] and use["idx"] == [0, 2, 3, 5] and use["pos"] == [0, 2, 3, 1] and use["has_prev"] == [False, True, True, True] and seen == [0, 1, 2, 3, 4, 5], f"{label}: dump_frames {use} {seen}"
    assert use["d2_none"][0] == 0.0 and use["d2_none"][1] > 0 and all(len(v) == 4 for v in use.values()) and set(use) >= set(E.JP.SUMS) | {"idx", "has_prev", "pos", "shift", "state_ss", "state_n"}
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
    mk = lambda keeps, links, d2, x2, pos, hp_, d2o, no_: dict(meta=dict(run="r", split="val", idx=list(range(len(pos))), use=dict(d2_none=d2, x2=x2, n=[int(h) for h in hp_], d2_other=d2o, x2_other=[x if k else 0.0 for x, k in zip(x2, no_)], n_other=no_, pos=pos, has_prev=hp_,
                                                                                                                                                           state_ss=[1.0] * len(pos), state_n=[1] * len(pos))), frames=[dict(token=f"t{i}", keep=np.array(k), rec=dict(src=np.array([a for a, _ in l_], int), dst=np.array([b for _, b in l_], int))) for i, (k, l_) in enumerate(zip(keeps, links))])  # noqa: E731
    s_ = mk([[1, 1, 1], [1, 0, 0], [1, 0, 0]], [[(0, 1), (1, 2)], [(0, 1)], [(0, 1)]], [0.0, 1.0, 6.0], [4.0, 4.0, 16.0], [0, 1, 2], [False, True, True], [0.0, 9.0, 0.0], [0, 2, 0])      # frame 2 has a past but no other scene's past
    o_ = mk([[1, 0, 1], [1, 0, 0], [1, 0, 0]], [[(0, 1)], [(0, 1)], [(0, 1)]], [0.0, 0.0, 0.0], [4.0, 4.0, 4.0], [0, 0, 0], [False, False, False], [0.0] * 3, [0] * 3)
    rep = E.use_report(s_, o_)
    assert abs(rep["u"] - 1.5) < 1e-9 and abs(rep["u_none"] - math.sqrt(7 / 20)) < 1e-9 and rep["frames_with_past"] == 2 and rep["frames_with_other_past"] == 1, rep
    assert abs(rep["u_by_position"]["1-1"] - 1.5) < 1e-9 and np.isnan(rep["u_by_position"]["2-2"]) and abs(rep["u_none_by_position"]["1-1"] - 0.5) < 1e-9 and abs(rep["u_none_by_position"]["2-2"] - math.sqrt(6 / 16)) < 1e-9 and np.isnan(rep["u_none_by_position"]["3-3"]), rep
    assert rep["against_off"] == dict(kept_gained_per_frame=1 / 3, kept_lost_per_frame=0.0, kept_flips_per_frame=1 / 3, frames_with_a_flip=1 / 3, links_changed_per_frame=1 / 3), rep["against_off"]
    m2 = dict(meta=dict(run="r", split="val", idx=[10, 11, 12, 13, 14], use=dict(d2_none=[0.0, 1.0, 4.0, 1.0, 1.0], x2=[4.0] * 5, n=[0, 1, 1, 1, 1], d2_other=[0.0] * 5, x2_other=[0.0] * 5, n_other=[0] * 5, pos=[0, 1, 5, 4, 1], has_prev=[False, True, True, True, True], state_ss=[8.0, 18.0, 50.0, 18.0, 36.0], state_n=[2] * 5)), frames=[])
    r2 = E.use_report(m2)                                                                                    # the carried state's RMS per stream position and the first-5 split, by hand
    sr = r2["state_rms_by_position"]
    assert len(sr) == 6 and sr[0] == 2.0 and abs(sr[1] - math.sqrt(54 / 4)) < 1e-12 and sr[4] == 3.0 and sr[5] == 5.0 and all(np.isnan(sr[q]) for q in (2, 3)), sr
    assert r2["position_per_frame"] == [0, 1, 5, 4, 1] and r2["frame_idx_first5"] == [10, 11, 13, 14] and r2["frame_idx_later"] == [12], r2
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
        assert [os.path.relpath(p, tmp) for p in paths.values()] == ["x/token_val_n12.pkl", "x_off/token_val_n12.pkl", "x_truncated/token_val_n12.pkl"], paths
        assert all(open(os.path.join(tmp, f"x_{md}", "model.pt"), "rb").read() == open(os.path.join(tmp, "x", "model.pt"), "rb").read() for md in ("off", "truncated")), "the sibling folder holds a copy of model.pt"
        refuses(lambda: E.dump(argparse.Namespace(run="x", seed=0, split="val", mode="stream", window=3, cpu=True, max=12), runs=tmp), "never overwritten")
        refuses(lambda: E.dump(argparse.Namespace(run="x", seed=1, split="val", mode="stream", window=3, cpu=True, max=5), runs=tmp), "does not match the run")
        d = {k: pickle.load(open(p, "rb")) for k, p in paths.items()}
        meta = d["stream"]["meta"]
        assert {"run", "seed", "split", "drop", "drop_seed", "idx", "every", "tokens", "K", "ident", "cache", "steps", "model"} <= set(meta) and meta["idx"] == list(range(12)) and meta["mode"] == "stream" and meta["every"] == 1 and meta["drop"] == 0.0 and meta["run"] == "x" and meta["model"] == "model.pt"
        assert d["off"]["meta"]["run"] == "x_off" and d["off"]["meta"]["of_run"] == "x" and d["truncated"]["meta"]["run"] == "x_truncated", "the meta names the sibling run the scorer reads"
        for sib in ("x", "x_off"):                                                                          # the scorer's own reader accepts the dumps: the cache, the kept nodes, the scorer's frames (a mode dump is read as a run of its own)
            r_ = E.JE.read_dump(argparse.Namespace(run=sib, seed=0, split="val", model="model.pt", every=1, drop=0.0, drop_seed=0), tmp, 12)
            assert len(r_["fr"]) == 12 and r_["idx"].tolist() == list(range(12)), sib
        assert [f["token"] for f in d["stream"]["frames"]] == meta["tokens"] and len(d["stream"]["frames"]) == 12 and meta["use"]["has_prev"][0] is False
        import lane_ptok_data as LP
        ds = LP.PTokData("val", 0, "e292_a")
        pred = E.JE.frame_prediction(ds[3]["tokens_np"], d["stream"]["frames"][3], "rec")
        assert pred["assign"] is None and pred["maps"] is None
        assert all(np.array_equal(d["stream"]["frames"][0][k], d["off"]["frames"][0][k]) for k in ("pnone", "keep")), "frame 0 has no past: stream = off"
        assert not np.array_equal(d["stream"]["frames"][3]["pnone"], d["off"]["frames"][3]["pnone"]), "the past changed nothing at frame 3"
        assert np.array_equal(d["truncated"]["frames"][2]["pnone"], d["stream"]["frames"][2]["pnone"]) and not np.array_equal(d["truncated"]["frames"][3]["pnone"], d["stream"]["frames"][3]["pnone"]), "truncated: window 3"
        rep = E.use_report(d["stream"], d["off"])
        assert rep["frames_with_past"] == 11 and rep["frames_with_other_past"] == 11 and rep["u"] > 0 and rep["u_none"] > 0 and rep["u"] != rep["u_none"] and rep["against_off"]["kept_flips_per_frame"] >= 0
        res = E.use(argparse.Namespace(run="x", split="val", max=12), runs=tmp)                              # the command: stream + the sibling x_off -> past_use_val_n12.json
        assert res["against_off"] == rep["against_off"] and res["against_off"]["kept_flips_per_frame"] > 0, "use reads the off dump of the sibling folder"
        assert os.path.exists(os.path.join(tmp, "x", "past_use_val_n12.json")) and res["position_per_frame"] == list(range(12)) and res["frame_idx_first5"] == [0, 1, 2, 3, 4] and res["frame_idx_later"] == list(range(5, 12)) and "against_off" in res
        assert len(res["state_rms_by_position"]) == 12 and all(v > 0 for v in res["state_rms_by_position"]), res["state_rms_by_position"]
        refuses(lambda: E.use(argparse.Namespace(run="x", split="val", max=12), runs=tmp), "never overwritten")
        with open(os.path.join(tmp, "x_off", "model.pt"), "ab") as fh:                                       # a sibling folder holds the same model.pt as the run or the dump is refused
            fh.write(b"0")
        refuses(lambda: E.dump(argparse.Namespace(run="x", seed=0, split="val", mode="off", window=3, cpu=True, max=5), runs=tmp), "is not a copy")


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


def real_chain_check(train_path=TRAIN):
    """one batch of the real TRAIN chains (the cache and its extension, 2 sequences of 3 keyframes) through the loop with the real head on the CPU, the sequence order of two arms with the same seeds, and `train()` with the real loader (3 steps, the stub head):
    the arguments it hands to `lane_past_data.loader` (the smoke path never calls it). Not part of the selftest: it needs the cache"""
    import json
    import torch
    import trainer as TR
    import lane_past_data as LPD
    T = load(train_path, "jpast_train_real")
    ds, dl = LPD.loader("train", 0, 2, 0)
    TR.seed_all(0)
    b = next(iter(dl))
    rows = torch.stack([f["row"] for f in b["frames"]], 1)
    assert tuple(rows.shape) == (2, 3) and (rows[:, 1:] - rows[:, :-1] == 1).all() and b["has_prev"][:, 1:].all() and tuple(b["rel_t"].shape) == (2, 3, 2) and b["frames"][0]["tok"].shape[1] == ds.tok.K, "real chains"
    torch.manual_seed(0)
    m = T.build_head(dict(no_past=False))
    for p in (m.past_block.o.weight, m.past_block.o.bias):
        torch.nn.init.normal_(p, std=0.3)
    m.train()
    m.set_step(0)
    loss, infos = T.sequence_loss(m, b, "cpu")
    loss.backward()
    assert torch.isfinite(loss) and len(infos) == 3 and all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in m.past_block.parameters()), "the real chains through the loop"

    def first(seed):
        TR.seed_all(seed)
        return [bb["idx"].tolist() for _, bb in zip(range(3), LPD.loader("train", 0, 4, 0)[1])]
    assert first(0) == first(0) and first(0) != first(1), "the sequence order follows the seed"
    calls, orig = [], LPD.loader

    def recording(*a, **k):
        calls.append((a, k))
        return orig(*a, **k)
    LPD.loader = recording
    try:
        with Patch(make_stub()), tempfile.TemporaryDirectory() as tmp:
            T.train(T.parse_args(["--name", "rl", "--seed", "1", "--init-seed", "0", "--bs", "2", "--seq", "2", "--steps", "3", "--workers", "0"]), runs=tmp, ckpt_every=500)
            ms, args = T.load_model("rl", "cpu", tmp)
            assert ms._step_py == 3 and args["seed"] == 1 and json.load(open(os.path.join(tmp, "rl", "config.json")))["seed"] == 0 and os.path.exists(os.path.join(tmp, "rl", ".done"))
    finally:
        LPD.loader = orig
    assert len(calls) == 1 and calls[0][0] == ("train", 1, 2, 0) and calls[0][1] == dict(K_seq=2, ident="e292_a", K=None), f"the loader call of train(): {calls}"
    print(f"real chains ok: loss {float(loss):.3f}, {len(ds)} chains, first batch rows {rows.tolist()}")


# ------------------------------------------------------------------------------------------------------------------------------ mutants
def one(args):
    k, (mod, name, a, b, *rest), tmp = args
    cmd = rest[0] if rest else "one"                                                                        # the command a mutant runs: the selftest ("one") or the real-data check ("real")
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
    r = subprocess.run([sys.executable, os.path.abspath(__file__), cmd, other[0], other[1]], capture_output=True, text=True, env=ENV)
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
    if len(sys.argv) > 1 and sys.argv[1] == "real":
        real_chain_check(sys.argv[2] if len(sys.argv) > 2 else TRAIN)
        sys.exit(0)
    for args_ in (["one", TRAIN, EVAL], ["real"]):
        r = subprocess.run([sys.executable, os.path.abspath(__file__)] + args_, capture_output=True, text=True, env=ENV)
        assert r.returncode == 0, f"{args_[0]} failed:\n{r.stdout[-1500:]}{r.stderr[-2500:]}"
        print(r.stdout.strip().splitlines()[-1])
    if len(sys.argv) < 2 or sys.argv[1] != "selftests":
        left = run_mutants()
        assert not left, f"mutants survived: {left}"
    print("test_jpast_train ok")
