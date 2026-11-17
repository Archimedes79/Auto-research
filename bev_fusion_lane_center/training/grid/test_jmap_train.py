"""Tests of the E374 trainer (`train/jmap_train.py`) and the read-out (`evaluation/jmap_eval.py`).

    wsl bash tools/env/_py14.sh test_jmap_train.py            # the selftest (a stub head with the S3 interface and the real `jtoken.JTokenHead`), the real-data loader check, then the hand mutants of both files: every mutant must make it FAIL
    wsl bash tools/env/_py14.sh test_jmap_train.py selftests  # the selftest and the real-data loader check only

The loop by hand: the maps of the batch reach the hook untouched, the head's counter advances once per optimiser step, the zero-initialised block = arm z, no GT key in the head's outputs, `v` / `v_none` logged as the mean of the last 100 steps (a block that ignores the map content: v = 0, v_none > 0), the smoke run with its resume, the options,
the hook check, the arms start from the same head, THE FIRST 20 STEPS OF ARM Z EQUAL `jtoken_train.py`'s (the loss of every step and the final weights, same batches). The read-out: the head frame by frame on the maps of the batch, the donor of the kill measure, the sums by hand, the dump of 12 real VAL frames in every map mode (a toy untrained head: the writer of
`jtoken_eval` accepts it, the modes differ, the siblings hold a copy of model.pt), the use report. The real-data loader check (`real`, needs the cache and the lane GT store): the batch of arm z is `lane_ptok_data`'s, the maps are those of `lane_map_data`, `train()` hands the right arguments to the loader. A mutant = one textual
change of the source of `jmap_train.py` or `jmap_eval.py`; the mutated copy is loaded in place of the file, four at a time.
"""
import argparse
import concurrent.futures as cf
import copy
import importlib.util
import inspect
import json
import math
import os
import pickle
import subprocess
import sys
import tempfile

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAIN, EVAL = os.path.join(ROOT, "training", "train", "jmap_train.py"), os.path.join(ROOT, "evaluation", "jmap_eval.py")
ENV = dict(os.environ, BFLC_ROOT=ROOT, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")      # a shared box: one thread per test process
for sub_ in ("training", "training/grid", "training/data", "training/data/prep", "training/train", "evaluation"):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))

T0, E0 = "jmap_train.py", "jmap_eval.py"
MUTANTS = [
    (T0, "the block is built before the head", "    head = JKT.build_head({})\n    if not args.get(\"no_map\"):\n        check_hook(head)\n        head.map_block = JM.MapBlock(d=JKT.arch({})[0], heads=JKT.arch({})[1])\n    return head",
     "    blk = JM.MapBlock(d=JKT.arch({})[0], heads=JKT.arch({})[1]) if not args.get(\"no_map\") else None\n    head = JKT.build_head({})\n    if blk is not None:\n        check_hook(head)\n        head.map_block = blk\n    return head"),
    (T0, "arm z has the block", "    if not args.get(\"no_map\"):\n        check_hook(head)", "    if True:\n        check_hook(head)"),
    (T0, "the hook is not checked", "    assert \"maps\" in inspect.signature(model.forward).parameters and hasattr(model, \"map_block\"), \\\n", "    assert True or \"maps\" in inspect.signature(model.forward).parameters and hasattr(model, \"map_block\"), \\\n"),
    (T0, "the maps are not handed to the head", "        return self.m(b, maps=self.maps)", "        return self.m(b)"),
    (T0, "the maps of the batch are not the ones handed over", "MapsView(m, {k: v.to(dev, non_blocking=True) for k, v in b[\"maps\"].items()})", "MapsView(m, {k: v.to(dev, non_blocking=True).roll(1, 0) for k, v in b[\"maps\"].items()})"),
    (T0, "arm z reads the maps", "        if no_map:\n            return base(m, b)", "        if no_map:\n            return base(MapsView(m, {k: v for k, v in b[\"maps\"].items()}), b)"),
    (T0, "v is not logged", "        info[\"v\"], info[\"v_none\"] = (float(np.mean(h)) if h else 0.0 for h in (v_hist, vn_hist))", "        pass"),
    (T0, "v is the last step's only", "(float(np.mean(h)) if h else 0.0 for h in (v_hist, vn_hist))", "(float(h[-1]) if h else 0.0 for h in (v_hist, vn_hist))"),
    (T0, "v is logged as v_none", "(float(np.mean(h)) if h else 0.0 for h in (v_hist, vn_hist))", "(float(np.mean(h)) if h else 0.0 for h in (v_hist, v_hist))"),
    (T0, "v takes the none-key value", 'v_hist.append(u["u"])', 'v_hist.append(u["u_none"])'),
    (T0, "the window of v_none is 10", "collections.deque(maxlen=V_WINDOW), collections.deque(maxlen=V_WINDOW)", "collections.deque(maxlen=V_WINDOW), collections.deque(maxlen=10)"),
    (T0, "V_WINDOW is 10", "V_WINDOW = 100 ", "V_WINDOW = 10 "),
    (T0, "the sampler order uses the cache seed", "seed=init, args=args,", "seed=a.seed, args=args,"),
    (T0, "the smoke batches use the cache seed", "init * 100003 + step, a.no_map, a.map_mode == \"other\")", "a.seed * 100003 + step, a.no_map, a.map_mode == \"other\")"),
    (T0, "another recipe is not refused", "assert c.get(\"design\") == \"jmap\" and old ==", "assert c.get(\"design\") == \"jmap\" or old =="),
    (T0, "the seed twin shares the head initialisation", "    init = a.seed if a.init_seed is None else a.init_seed\n", "    init = a.seed\n"),
    (T0, "no_map is not in the recipe", 'RECIPE = ("seed", "init_seed", "cache", "ident", "K", "bs", "steps", "lr", "no_map", "map_mode")', 'RECIPE = ("seed", "init_seed", "cache", "ident", "K", "bs", "steps", "lr", "map_mode")'),
    (T0, "arm m with one frame per step is accepted", 'assert args["no_map"] or args["bs"] >= 2,', 'assert True or args["bs"] >= 2,'),
    (T0, "the loaded head has no block", "    m = build_head(ck[\"args\"])\n    m.load_state_dict(ck[\"state\"])", "    m = build_head(dict(ck[\"args\"], no_map=True))\n    m.load_state_dict(ck[\"state\"], strict=False)"),
    (T0, "the smoke maps lie on another frame's tokens", 'pos=(b["pos"][f, idx].numpy()', 'pos=(b["pos"][f - 1, idx].numpy()'),
    (T0, "the loader gets the cache seed as the init seed", "init_seed=init, maps=not a.no_map", "init_seed=a.seed, maps=not a.no_map", "real"),
    (T0, "the loader builds the maps of arm z", "maps=not a.no_map, mode=a.map_mode, ident=a.ident", "maps=True, mode=a.map_mode, ident=a.ident", "real"),
    (T0, "the capacity control loads the maps of its own frame", "maps=not a.no_map, mode=a.map_mode, ident=a.ident", 'maps=not a.no_map, mode="train", ident=a.ident', "real"),
    (T0, "map_mode is not in the recipe", '"lr", "no_map", "map_mode")', '"lr", "no_map")'),
    (T0, "the capacity control without the block is accepted", 'assert not args["no_map"] or args["map_mode"] == "train",', 'assert True or args["map_mode"] == "train",'),
    (T0, "the smoke maps of the capacity control are the frame's own", "maps={k: v.roll(1, 0) for k, v in maps.items()} if other else maps)", "maps=maps)"),
    (T0, "the smoke batches ignore the map mode", 'init * 100003 + step, a.no_map, a.map_mode == "other")', "init * 100003 + step, a.no_map, False)"),
    (T0, "the maps stay on the CPU", 'MapsView(m, {k: v.to(dev, non_blocking=True) for k, v in b["maps"].items()})', 'MapsView(m, {k: v for k, v in b["maps"].items()})'),
    (T0, "the checkpoint interval is 400 steps", "ckpt_every=500, smoke_k=SMOKE_K):", "ckpt_every=400, smoke_k=SMOKE_K):"),
    (T0, "the smoke frames have 384 tokens", "SMOKE_K = 512 ", "SMOKE_K = 384 "),
    (T0, "the loader gets another batch size", "\"train\", a.seed, a.bs, a.workers", "\"train\", a.seed, 4, a.workers", "real"),
    (T0, "the loader gets the head seed as the cache seed", "\"train\", a.seed, a.bs, a.workers", "\"train\", init, a.bs, a.workers", "real"),
    (E0, "every mode goes to a sibling", 'return run if mode == RECORD else f"{run}_{mode}"', 'return f"{run}_{mode}"'),
    (E0, "the record mode is clean", 'RECORD = "err"', 'RECORD = "clean"'),
    (E0, "err reads the exact map", 'LOADER_MODE = dict(err="error", clean="exact",', 'LOADER_MODE = dict(err="exact", clean="error",'),
    (E0, "U and C2 are swapped", 'S="S", U="U", C2="C2")', 'S="S", U="C2", C2="U")'),
    (E0, "the TRAIN read-out takes the VAL frames", "idx = JE.sample(len(ds), a.split, 1)[: a.max or None]", 'idx = JE.sample(len(ds), "val", 1)[: a.max or None]'),
    (E0, "params counts the block twice", "params_block=JK.count_params(m.map_block) if getattr(m, \"map_block\", None) is not None else 0)", "params_block=2 * JK.count_params(m.map_block) if getattr(m, \"map_block\", None) is not None else 0)"),
    (E0, "the GMACs without the block are those with it", "gmacs_no_map = head_macs(m, first) if not no_map else gmacs", "gmacs_no_map = gmacs"),
    (E0, "the timing ignores --K", "K = a.K or ds.tok.K", "K = ds.tok.K"),
    (E0, "the timing reads the first frames of a shorter list", "items = [cut(LMD.collate([ds[i]])) for i in range(a.frames)]", "items = [cut(LMD.collate([ds[i]])) for i in range(a.frames + 1)]"),
    (E0, "the timing measures nothing", "median_ms=float(np.median(flat))", "median_ms=0.0"),
    (E0, "the sibling has no model.pt", "        JPE.sibling(runs, a.run, a.mode)", "        pass"),
    (E0, "the meta names the run, not the sibling", "meta = dict(run=name,", "meta = dict(run=a.run,"),
    (E0, "dump does not check the seed", 'assert args["seed"] == a.seed,', 'assert True or args["seed"] == a.seed,'),
    (E0, "the donor is ignored", 'meter.donor = to_device(b["donor"], dev)', "meter.donor = None"),
    (E0, "the donor is the frame's own map", 'meter.donor = to_device(b["donor"], dev)', 'meter.donor = to_device(b["maps"], dev)'),
    (E0, "the head gets the donor as its map", 'out = m(frd, maps=maps) if block is not None else m(frd)', 'out = m(frd, maps=to_device(b["donor"], dev)) if block is not None else m(frd)'),
    (E0, "the sums are not cleared", "meter.take_sums() if meter is not None", "meter.acc if meter is not None"),
    (E0, "n_map counts the padding", 'int(b["maps"]["valid"].sum()) if block is not None else 0', 'int(b["maps"]["valid"].numel()) if block is not None else 0'),
    (E0, "a batch of two frames", "batch_size=1, shuffle=False", "batch_size=2, shuffle=False"),
    (E0, "use: v is taken from the none-key sums", 'v=ratio(u["d2_other"], u["x2_other"], ok_o)', 'v=ratio(u["d2_none"], u["x2"], ok_n)'),
    (E0, "use: v_none is taken from the other-sample sums", 'v_none=ratio(u["d2_none"], u["x2"], ok_n)', 'v_none=ratio(u["d2_other"], u["x2_other"], ok_o)'),
    (E0, "use: v by frame is the overall one", "v_frame=per(u[\"d2_other\"], u[\"x2_other\"], ok_o)", "v_frame=[ratio(u[\"d2_other\"], u[\"x2_other\"], ok_o)] * len(u[\"n\"])"),
    (E0, "use: the frames without a donor are not NaN", "ok_o, ok_n = u[\"n_other\"] > 0, u[\"n\"] > 0", "ok_o, ok_n = u[\"n_other\"] >= 0, u[\"n\"] > 0"),
    (E0, "use reads the sibling's dump", "d = pickle.load(open(JPE.dump_file(runs, a.run, a.split, a.max), \"rb\"))", "d = pickle.load(open(JPE.dump_file(runs, a.run + \"_none\", a.split, a.max), \"rb\"))"),
]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def same_t(a, b):
    """two tensors are equal, NaN = NaN (the yscale of a batch is NaN outside the long GT links)"""
    import torch
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(torch.nan_to_num(a.double(), nan=-7.5), torch.nan_to_num(b.double(), nan=-7.5))


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
        """the S3 interface of `jtoken.JTokenHead` with a tiny body: forward(batch, past=None, maps=None); after one projection `map_block(x, pos, dirv, valid, maps)` when both exist; the output dict that `jtoken_loss` reads + x, w; set_step / _step_py / the step buffer"""

        def __init__(self, d=64):
            super().__init__()
            self.proj, self.pe, self.types, self.q, self.k, self.none, self.sc = nn.Linear(64, d), nn.Linear(4, d), nn.Linear(d, 7), nn.Linear(d, 8), nn.Linear(d, 8), nn.Linear(d, 2), nn.Linear(d, 2)
            self.register_buffer("step", torch.zeros((), dtype=torch.long))
            self._step_py = 0
            self.map_block = None

        def set_step(self, n):
            self.step.fill_(n)
            self._step_py = int(n)

        def _load_from_state_dict(self, *a, **k):
            super()._load_from_state_dict(*a, **k)
            self._step_py = int(self.step)

        def forward(self, batch, past=None, maps=None):
            valid, pos, dirv = batch["valid"].bool(), batch["pos"].float(), batch["dirv"].float()
            x = self.proj(batch["tok"].float()) + self.pe(torch.cat([pos / 50.0, dirv], -1))
            if self.map_block is not None and maps is not None:
                x = self.map_block(x, pos, dirv, valid, maps)
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
    return "maps" in inspect.signature(JK.JTokenHead.forward).parameters and "map_block" in inspect.signature(JK.JTokenHead.__init__).parameters


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
    """record every forward of the head: the step counter before it, the maps and the output; `m.forward` is replaced on the instance (`restore()` puts it back)"""
    orig, calls = m.forward, []

    def forward(batch, past=None, maps=None):
        c = dict(step=m._step_py, maps=maps, batch=batch)
        out = orig(batch, past, maps)
        calls.append(dict(c, out=out))
        return out
    m.forward = forward
    return calls, lambda: m.__dict__.pop("forward")


class NoRun:
    def log(self, *a, **k):
        pass


def live(blk, std=0.3, seed=0):
    """a block with a non-zero output projection (a trained-like one)"""
    import torch
    g = torch.Generator().manual_seed(seed)
    for p in (blk.o.weight, blk.o.bias):
        p.data.copy_(torch.randn(p.shape, generator=g) * std)
    return blk


def blind(blk):
    """a block that ignores the map content: q = 0, geometry bias 0, v = the constant 1 for every key, v_none = 0, o = identity (`jmap.selftest`'s critic case): v = 0, v_none > 0"""
    import torch
    with torch.no_grad():
        for p_ in (blk.q.weight, blk.q.bias, blk.bias[2].weight, blk.bias[2].bias, blk.v_none, blk.kv.weight, blk.kv.bias):
            p_.zero_()
        blk.kv.bias[64:] = 1.0
        blk.o.weight.copy_(torch.eye(64))
        blk.o.bias.zero_()
    return blk


# ------------------------------------------------------------------------------------------------------------------------------ the trainer
def trainer_tests(T, stub, label):
    import numpy as np
    import torch
    import jmap as JM
    import jtoken as JK
    import lane_map_data as LMD
    B, K = 4, 48
    # the arms start from the same head: the block is built after it
    torch.manual_seed(5)
    hz = T.build_head(dict(no_map=True))
    torch.manual_seed(5)
    hm = T.build_head(dict(no_map=False))
    sz, sm = hz.state_dict(), hm.state_dict()
    assert hz.map_block is None and isinstance(hm.map_block, JM.MapBlock) and set(sm) - set(sz) == {k for k in sm if k.startswith("map_block.")} and all(torch.equal(sz[k], sm[k]) for k in sz), f"{label}: the arms do not start from the same head"
    # the smoke batch: arm z's is `jtoken.synthetic_batch`, arm m's adds the maps (the fixed key set, near the lane tokens, valid samples in every frame)
    bz, bm = T.synthetic_batch(B, K, 1, True), T.synthetic_batch(B, K, 1)
    jb = JK.synthetic_batch(B, K, seed=1)
    assert sorted(bz) == sorted(jb) and all(same_t(bz[k], jb[k]) for k in jb), f"{label}: the batch of arm z is not jtoken's"
    assert sorted(set(bm) - set(bz)) == ["maps"] and sorted(bm["maps"]) == sorted(LMD.KEYS) and all(same_t(bm[k], bz[k]) for k in bz) and bm["maps"]["valid"].sum(1).min() > 0 and bm["maps"]["pos"].shape == (B, T.SMOKE_M, 2) and T.SMOKE_M == 704, f"{label}: the smoke maps (the cap)"
    d_ = torch.cdist(bm["maps"]["pos"], bm["pos"])                                                      # every smoke map sample lies within 2.5 m of a token of ITS frame (the lane tokens, 0.5 m of noise)
    assert (d_.min(-1).values[bm["maps"]["valid"]] < 2.5).all(), f"{label}: the smoke maps are not near the tokens of their frames"
    # arm z: the E365 loss, the counter once per step, no v
    hz.train()
    hz.set_step(0)
    loss_z, info_z = T.make_loss(NoRun(), hz, True)(hz, bz)
    hz.set_step(0)
    want, _ = JK.jtoken_loss(hz(bz), bz)
    assert float(loss_z) == float(want) and "v" not in info_z and "v_none" not in info_z and hz._step_py == 1 and int(hz.step) == 1, f"{label}: arm z is the E365 loss, one counter step per step"
    # the zero-initialised block = arm z on the same weights (bit for bit: x + 0); v and v_none are 0
    hm.train()
    hm.set_step(0)
    hz.set_step(0)
    lm = T.make_loss(NoRun(), hm, False)
    loss_m0, info0 = lm(hm, bm)
    loss_z2, _ = T.make_loss(NoRun(), hz, True)(hz, bm)
    assert float(loss_m0) == float(loss_z2) and info0["v"] == 0.0 and info0["v_none"] == 0.0 and "nll_s" in info0 and "gpu_gb" in info0, f"{label}: the zero-initialised block changed the loss"
    # a live block: the maps of the batch reach the hook untouched (no GT key among them), the counter is advanced once, the loss changes, every parameter of the block gets a gradient
    live(hm.map_block)
    calls, restore = spy(hm)
    hm.set_step(0)
    loss_m, _ = lm(hm, bm)
    restore()
    assert len(calls) == 1 and calls[0]["step"] == 0 and hm._step_py == 1 and int(hm.step) == 1, f"{label}: one forward and one counter step per optimiser step"
    assert sorted(calls[0]["maps"]) == sorted(LMD.KEYS) and all(torch.equal(calls[0]["maps"][k], bm["maps"][k]) for k in LMD.KEYS), f"{label}: the maps of the batch are not the ones the hook gets"
    assert abs(float(loss_m) - float(loss_z2)) > 1e-6, f"{label}: a live block did not change the loss"
    loss_m.backward()
    dead = [n for n, p in hm.map_block.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum() == 0]
    assert not dead, f"{label}: the loss does not reach the block: {dead}"
    # nothing of the GT enters the output: the GT labels of the batch zeroed, the outputs of the head do not change
    b2 = copy.deepcopy(bm)
    b2["y"], b2["ytype"], b2["ylong"], b2["yscale"] = torch.zeros_like(b2["y"]), torch.zeros_like(b2["ytype"]), torch.zeros_like(b2["ylong"]), torch.zeros_like(b2["yscale"])
    calls2, restore = spy(hm)
    hm.set_step(0)
    lm(hm, b2)
    restore()
    assert torch.equal(calls[0]["out"]["logp"], calls2[0]["out"]["logp"]) and torch.equal(calls[0]["out"]["x"], calls2[0]["out"]["x"]), f"{label}: a GT label changed an output"
    # v and v_none: the block that ignores the map content has v = 0 and v_none > 0 (the critic's case through the trainer), a reading block has both > 0; the means of the last steps
    torch.manual_seed(6)
    hb = T.build_head(dict(no_map=False))
    hb.train()
    blind(hb.map_block)
    ib = T.make_loss(NoRun(), hb, False)(hb, bm)[1]
    assert ib["v"] < 1e-6 and ib["v_none"] > 0.01, f"{label}: a block that ignores the map content: v {ib['v']} v_none {ib['v_none']}"
    torch.manual_seed(6)
    h0 = T.build_head(dict(no_map=False))
    h0.train()
    live(h0.map_block)
    i1 = T.make_loss(NoRun(), h0, False)(h0, bm)[1]
    assert i1["v"] > 0 and i1["v_none"] > 0 and i1["v"] != i1["v_none"], f"{label}: v and v_none of a reading block"
    assert T.V_WINDOW == 100 == inspect.signature(T.T.fit).parameters["log_every"].default and 5 * T.V_WINDOW == 500, f"{label}: V_WINDOW {T.V_WINDOW}"
    h1 = T.build_head(dict(no_map=False))
    h1.train()
    live(h1.map_block)
    rec, orig = [], T.JM.UseMeter.take

    def take(self):
        r = orig(self)
        rec.append((r["u"], r["u_none"]) if r else None)
        return r
    T.JM.UseMeter.take = take
    try:
        lf1 = T.make_loss(NoRun(), h1, False)
        for _ in range(120):
            with torch.no_grad():
                h1.map_block.o.weight.mul_(1.02)                                                       # v grows from step to step: the window can be seen
            info1 = lf1(h1, bm)[1]
    finally:
        T.JM.UseMeter.take = orig
    for j_, key in ((0, "v"), (1, "v_none")):
        vals = [r_[j_] for r_ in rec]
        assert abs(info1[key] - float(np.mean(vals[-100:]))) < 1e-9 and abs(info1[key] - float(np.mean(vals[-10:]))) > 1e-4, f"{label}: {key} is not the mean of the last 100 steps"
    # the capacity control: the smoke batch of mode other gives every frame the map of ANOTHER frame of the batch (rolled by one): the maps are no longer near the tokens of their frame; everything else is the same batch
    bo = T.synthetic_batch(B, K, 1, False, True)
    assert all(torch.equal(bo["maps"][k], bm["maps"][k].roll(1, 0)) for k in LMD.KEYS) and all(same_t(bo[k], bm[k]) for k in bm if k != "maps"), f"{label}: the smoke maps of mode other"
    d_o = torch.cdist(bo["maps"]["pos"], bo["pos"]).min(-1).values[bo["maps"]["valid"]]
    assert (d_o < 2.5).float().mean() < 0.5 and (d_.min(-1).values[bm["maps"]["valid"]] < 2.5).all(), f"{label}: in mode other the maps are those of other frames"
    # the maps are moved to the device of the head (a head whose first parameter lives on the `meta` device stands in for a GPU: the head sees where its maps went)
    class Stop(Exception):
        pass

    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dev = torch.nn.Parameter(torch.empty(1, device="meta"))
            self.map_block = JM.MapBlock()
            self.seen = None

        def forward(self, batch, past=None, maps=None):
            self.seen = {k: v.device.type for k, v in maps.items()}
            raise Stop
    pr = Probe()
    try:
        T.make_loss(NoRun(), pr, False)(pr, bm)
        raise AssertionError("the probe head was not called")
    except Stop:
        pass
    assert pr.seen == dict(pos="meta", dir="meta", count="meta", valid="meta"), f"{label}: the maps are not moved to the device of the head: {pr.seen}"
    # the smoke run to its end: files, the counter of the head = the steps, then the same run after a kill = the straight run
    with tempfile.TemporaryDirectory() as tmp:
        a = T.parse_args(["--name", "m", "--smoke", "--bs", "2", "--seed", "0"])
        T.train(a, runs=tmp, ckpt_every=10, smoke_k=K)
        for f in (".done", "model.pt", "config.json", "log.jsonl", "ckpt.pt"):
            assert os.path.exists(os.path.join(tmp, "m", f)), f"{label}: {f} missing"
        m, args = T.load_model("m", "cpu", tmp)
        assert m._step_py == 20 and int(m.step) == 20 and args["init_seed"] == 0 and isinstance(m.map_block, JM.MapBlock), f"{label}: the head counted {m._step_py} forwards for 20 steps"
        assert json.load(open(os.path.join(tmp, "m", "config.json")))["design"] == "jmap"
        rows = [json.loads(l_) for l_ in open(os.path.join(tmp, "m", "log.jsonl"))]
        assert "v" in rows[-1] and "v_none" in rows[-1] and "total" in rows[-1], f"{label}: v is in the rows of log.jsonl"
        T.train(a, runs=tmp, ckpt_every=10, smoke_k=K)                                                  # the same recipe in a finished folder is resumed (nothing left to do)
        cfg, folder = json.load(open(os.path.join(tmp, "m", "config.json")))["args"], os.path.join(tmp, "m")
        change = dict(seed=lambda v: v + 1, init_seed=lambda v: v + 1, cache=lambda v: "another_root", ident=lambda v: "e273_p", K=lambda v: 384, bs=lambda v: v + 1, steps=lambda v: v + 1, lr=lambda v: v * 2, no_map=lambda v: not v, map_mode=lambda v: "other" if v == "train" else "train")
        assert set(change) == set(T.RECIPE), f"{label}: a recipe key without a refusal test: {set(T.RECIPE) ^ set(change)}"
        T.check_start(cfg, folder)                                                                       # the same recipe is accepted
        for k_ in change:                                                                                # a change of EVERY recipe key is refused at a resume
            refuses(lambda: T.check_start(dict(cfg, **{k_: change[k_](cfg[k_])}), folder), "of another run")
        refuses(lambda: T.train(T.parse_args(["--name", "m", "--smoke", "--bs", "2", "--seed", "1"]), runs=tmp, ckpt_every=10, smoke_k=K), "of another run")
        real_make, state = T.make_loss, dict(boom=True)

        def flaky(run, model, no_map=False):
            f, n = real_make(run, model, no_map), [0]

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
        ia = T.parse_args(["--name", "i", "--smoke", "--bs", "2", "--seed", "0", "--init-seed", "3"])
        T.train(ia, runs=tmp, ckpt_every=10, smoke_k=K)
        mi, ai = T.load_model("i", "cpu", tmp)
        assert json.load(open(os.path.join(tmp, "i", "config.json")))["seed"] == 3 and ai["seed"] == 0 and ai["init_seed"] == 3 and json.load(open(os.path.join(tmp, "m", "config.json")))["seed"] == 0, f"{label}: --init-seed is the seed of the loop"
        za = T.parse_args(["--name", "z", "--smoke", "--bs", "2", "--seed", "0", "--no-map"])
        T.train(za, runs=tmp, ckpt_every=10, smoke_k=K)
        oa = T.parse_args(["--name", "o", "--smoke", "--bs", "2", "--seed", "0", "--map-mode", "other"])
        seen_b, real_fit = [], T.T.fit

        def fit_spy(model, loss_fn, run, **kw):                                                         # the smoke batch of step 3 as the loop asks for it
            seen_b.append(kw["batch_fn"](3))
            return real_fit(model, loss_fn, run, **kw)
        T.T.fit = fit_spy
        try:
            T.train(oa, runs=tmp, ckpt_every=10, smoke_k=K)
        finally:
            T.T.fit = real_fit
        want_b, plain_b = T.synthetic_batch(2, K, 3, False, True), T.synthetic_batch(2, K, 3, False, False)
        assert all(torch.equal(seen_b[0]["maps"][k], want_b["maps"][k]) for k in LMD.KEYS) and not torch.equal(seen_b[0]["maps"]["pos"], plain_b["maps"]["pos"]), f"{label}: the smoke batches of --map-mode other"
        mo, ao = T.load_model("o", "cpu", tmp)
        assert ao["map_mode"] == "other" and mo._step_py == 20 and isinstance(mo.map_block, JM.MapBlock), f"{label}: the capacity control run"
        refuses(lambda: T.train(a.__class__(**dict(vars(oa), name="m")), runs=tmp, ckpt_every=10, smoke_k=K), "of another run")                 # the folder of arm m is not the capacity control's
        mz, _ = T.load_model("z", "cpu", tmp)
        assert mz.map_block is None and mz._step_py == 20 and not any(k.startswith("map_block") for k in mz.state_dict()), f"{label}: arm z has a block"
    # the options: arm m needs two frames per step (v compares with another sample of the batch), arm z does not; the capacity control (--map-mode other) needs the block
    refuses(lambda: T.parse_args(["--name", "x", "--bs", "1"]), "at least 2 frames")
    T.parse_args(["--name", "x", "--bs", "1", "--no-map"])
    refuses(lambda: T.parse_args(["--name", "x", "--no-map", "--map-mode", "other"]), "needs the block")
    assert T.parse_args(["--name", "x", "--map-mode", "other"]).map_mode == "other" and T.parse_args(["--name", "x"]).map_mode == "train"
    # the defaults of the loop that a run uses (the tests pass their own): a checkpoint every 500 steps, the smoke frames of the real K
    sg = inspect.signature(T.train).parameters
    assert sg["ckpt_every"].default == 500 and sg["smoke_k"].default == T.SMOKE_K == 512, f"{label}: the defaults of train(): {sg['ckpt_every'].default} {sg['smoke_k'].default}"


def equals_jtoken_train(T, label):
    """THE FIRST 20 STEPS OF ARM Z ARE `jtoken_train.py`'s: the same synthetic batches, the same loss at every step, the same final weights (the card's pin); the head seed of `jtoken_train` (--head-seed) is the init seed here"""
    import torch
    import jtoken_train as JKT
    import trainer
    hist, real_fit = {}, trainer.fit

    def fit(model, loss_fn, run, **kw):
        h = real_fit(model, loss_fn, run, **kw)
        hist[run.dir] = list(h)
        return h
    trainer.fit = fit
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for seed, init in ((0, None), (0, 3)):
                nj, nz = ("j_default", "z_default") if init is None else (f"j{init}", f"z{init}")
                JKT.train(JKT.parse_args(["--name", nj, "--smoke", "--bs", "2", "--seed", str(seed)] + ([] if init is None else ["--head-seed", str(init)])), runs=tmp, ckpt_every=10, smoke_k=48)
                T.train(T.parse_args(["--name", nz, "--smoke", "--bs", "2", "--seed", str(seed), "--no-map"] + ([] if init is None else ["--init-seed", str(init)])), runs=tmp, ckpt_every=10, smoke_k=48)
                hj, hz = hist[os.path.join(tmp, nj)], hist[os.path.join(tmp, nz)]
                assert len(hj) == len(hz) == 20 and hj == hz, f"{label}: the loss of arm z at steps 1 - 20 is not jtoken_train's (init {init}): {hj[:3]} {hz[:3]}"
                sj, sz = JKT.load_model(nj, "cpu", tmp)[0].state_dict(), T.load_model(nz, "cpu", tmp)[0].state_dict()
                assert list(sj) == list(sz) and all(torch.equal(sj[k], sz[k]) for k in sj), f"{label}: the final weights of arm z are not jtoken_train's"
            assert hist[os.path.join(tmp, "j_default")] != hist[os.path.join(tmp, "j3")], f"{label}: another init seed gave the same run"
    finally:
        trainer.fit = real_fit


def hook_refusal(T):
    """a head without the S3 hook is refused for arm m and accepted for arm z"""
    import torch.nn as nn
    import jtoken_train as JKT

    class NoHook(nn.Module):
        def forward(self, batch):
            return {}
    old = JKT.build_head
    JKT.build_head = lambda args: NoHook()
    try:
        refuses(lambda: T.build_head(dict(no_map=False)), "S3 hook")
        assert isinstance(T.build_head(dict(no_map=True)), NoHook)
    finally:
        JKT.build_head = old


# ------------------------------------------------------------------------------------------------------------------------------ the read-out
def eval_tests(E, T, stub, label):
    import numpy as np
    import torch
    import jmap as JM
    import jtoken as JK
    K, n = 48, 6
    torch.manual_seed(1)
    m = T.build_head(dict(no_map=False))
    live(m.map_block)
    m.eval()
    frames = [dict(JK.with_tokens_np(JK.synthetic_batch(1, K, seed=10 + i)), token=[f"t{i}"], row=torch.tensor([i])) for i in range(n)]
    maps = [T.synthetic_maps(f, 100 + i) for i, f in enumerate(frames)]
    batches = [dict(f, maps=maps[i], donor=maps[(i + 1) % n]) for i, f in enumerate(frames)]            # the donor of frame i = the map of frame i + 1
    # the head frame by frame on the frame's own map, the counts by hand
    got = list(E.iter_outputs(m, batches, "cpu"))
    assert len(got) == n and [b["token"] for b, _, _ in got] == [[f"t{i}"] for i in range(n)]
    for i, (b, out, info) in enumerate(got):
        with torch.no_grad():
            want = m(frames[i], maps=maps[i])
        assert torch.equal(out["logp"], want["logp"]) and info["n_map"] == int(maps[i]["valid"].sum()) and info["n"] == int(frames[i]["valid"].sum()), f"{label}: frame {i}"
    assert not torch.equal(got[0][1]["logp"], m(frames[0], maps={})["logp"]), f"{label}: the map changed nothing"
    # the sums of the kill measures by hand from the block's own inputs: the update with the frame's map against the one with the donor (another scene's map) and against the none key only
    cap = {}
    hk = m.map_block.register_forward_hook(lambda mod, args, out: cap.__setitem__("args", args))
    try:
        for i, (b, out, info) in enumerate(E.iter_outputs(m, batches, "cpu")):
            xin, pi, di, vi, ma = cap["args"]
            with torch.no_grad():
                y_s, y_o, y_n = (m.map_block.delta(xin, pi, di, vi, q_) for q_ in (ma, maps[(i + 1) % n], None))
            mv = vi[..., None].float()
            want_o, want_n = float((((y_s - y_o) * mv) ** 2).sum()), float((((y_s - y_n) * mv) ** 2).sum())
            assert info["n_other"] == int(vi.sum()) == info["n"] and abs(info["d2_other"] - want_o) < 1e-4 * max(want_o, 1) and abs(info["d2_none"] - want_n) < 1e-4 * max(want_n, 1) and want_o > 0 and want_n > 0 and info["x2_other"] == info["x2"] > 0, f"{label}: the sums of frame {i}: {info} {want_o} {want_n}"
    finally:
        hk.remove()
    # dump_frames: every frame in order, the use arrays aligned, the dumped frame = the head's output
    fr_d, use = E.dump_frames(m, batches, "cpu", log=lambda *a, **k: None)
    assert [f["token"] for f in fr_d] == [f"t{i}" for i in range(n)] and set(use) == set(E.USE_KEYS) and all(len(v) == n for v in use.values()) and use["n_map"] == [int(mp["valid"].sum()) for mp in maps], f"{label}: dump_frames {use}"
    assert np.array_equal(fr_d[2]["pnone"], E.JE.batch_dump(got[2][1], frames[2])[0]["pnone"]), f"{label}: the dumped frame is the head's output"
    pred = E.JE.frame_prediction(frames[2]["tokens_np"][0], fr_d[2], "rec")                              # the writer of jtoken_eval accepts a dumped frame (no GT field)
    assert pred["assign"] is None and pred["maps"] is None
    # arm z: no block, no maps handed over, every frame the plain head on the frame
    mz = T.build_head(dict(no_map=True))
    mz.eval()
    for i, (b, out, info) in enumerate(E.iter_outputs(mz, [dict(fr, maps=maps[j], donor=maps[j]) for j, fr in enumerate(frames)], "cpu")):
        with torch.no_grad():
            assert torch.equal(out["logp"], mz(frames[i])["logp"]) and info["n"] == 0 and info["n_map"] == 0 and info["d2_none"] == 0, f"{label}: arm z frame {i}"
    # the names: the record mode is the run's own folder, every other mode a sibling; the read-out names map to the modes of the data module (a literal here)
    assert E.MODES == ("err", "clean", "gross", "none", "S", "U", "C2") and E.RECORD == "err" and E.run_folder("r", "err") == "r" and [E.run_folder("r", md) for md in E.MODES[1:]] == [f"r_{md}" for md in E.MODES[1:]]
    assert E.LOADER_MODE == dict(err="error", clean="exact", gross="gross", none="none", S="S", U="U", C2="C2"), E.LOADER_MODE
    # the use report by hand: v over the frames that have a donor, v_none over those with a map read, NaN per frame where undefined
    mk = lambda d2, x2, n_, d2o, x2o, no_, nm: dict(meta=dict(run="r", split="val", mode="err", use=dict(d2_none=d2, x2=x2, n=n_, d2_other=d2o, x2_other=x2o, n_other=no_, n_map=nm)))  # noqa: E731
    rep = E.use_report(mk([1.0, 4.0, 0.0], [4.0, 4.0, 0.0], [5, 5, 0], [9.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5, 0, 0], [30, 20, 0]))
    assert rep["frames"] == 3 and rep["frames_with_map"] == 2 and rep["frames_with_donor"] == 1 and abs(rep["v"] - 1.5) < 1e-12 and abs(rep["v_none"] - math.sqrt(5 / 8)) < 1e-12, rep
    assert abs(rep["v_frame"][0] - 1.5) < 1e-12 and np.isnan(rep["v_frame"][1]) and np.isnan(rep["v_frame"][2]) and abs(rep["v_none_frame"][0] - 0.5) < 1e-12 and abs(rep["v_none_frame"][1] - 1.0) < 1e-12 and np.isnan(rep["v_none_frame"][2]) and rep["n_map_frame"] == [30, 20, 0], rep


def dump_on_real_frames(E, T):
    """the dump command on 12 real VAL frames of the cache and the lane GT store with a toy head (the stub with a live block): names, meta, the writer of jtoken_eval, every map mode by hand, the siblings, arm z, `use`"""
    import numpy as np
    import torch
    import jtoken_train as JKT
    import lane_map_data as LMD
    import lane_ptok_data as LP
    with tempfile.TemporaryDirectory() as tmp:
        torch.manual_seed(2)
        m = T.build_head(dict(no_map=False))
        live(m.map_block)
        os.makedirs(os.path.join(tmp, "x"))
        args = dict(seed=0, ident="e292_a", K=None, cache=None, steps=12000, no_map=False, init_seed=0)
        torch.save(dict(state=m.state_dict(), args=args, steps=12000), os.path.join(tmp, "x", "model.pt"))
        ns = lambda mode, run="x", seed=0, mx=12, split="val": argparse.Namespace(run=run, seed=seed, split=split, mode=mode, cpu=True, max=mx)  # noqa: E731
        paths = {md: E.dump(ns(md), runs=tmp) for md in E.MODES}
        assert [os.path.relpath(p, tmp) for p in paths.values()] == ["x/token_val_n12.pkl"] + [f"x_{md}/token_val_n12.pkl" for md in E.MODES[1:]], paths
        assert all(open(os.path.join(tmp, f"x_{md}", "model.pt"), "rb").read() == open(os.path.join(tmp, "x", "model.pt"), "rb").read() for md in E.MODES[1:]), "the sibling folder holds a copy of model.pt"
        refuses(lambda: E.dump(ns("err"), runs=tmp), "never overwritten")
        refuses(lambda: E.dump(ns("err", seed=1, mx=5), runs=tmp), "does not match the run")
        d = {md: pickle.load(open(p, "rb")) for md, p in paths.items()}
        meta = d["err"]["meta"]
        assert {"run", "seed", "split", "drop", "drop_seed", "idx", "every", "tokens", "K", "ident", "cache", "steps", "model"} <= set(meta) and meta["idx"] == list(range(12)) and meta["mode"] == "err" and meta["every"] == 1 and meta["drop"] == 0.0 and meta["run"] == "x" and meta["model"] == "model.pt"
        assert all(d[md]["meta"]["run"] == f"x_{md}" and d[md]["meta"]["of_run"] == "x" and d[md]["meta"]["mode"] == md for md in E.MODES[1:])
        for sib in ["x"] + [f"x_{md}" for md in E.MODES[1:]]:                                               # the scorer's own reader accepts every dump (a mode dump is read as a run of its own)
            r_ = E.JE.read_dump(argparse.Namespace(run=sib, seed=0, split="val", model="model.pt", every=1, drop=0.0, drop_seed=0), tmp, 12)
            assert len(r_["fr"]) == 12 and r_["idx"].tolist() == list(range(12)), sib
        # every mode is the head on the map the mode names, by hand (a literal list: the mode -> the map)
        ds = LMD.MapData("val", 0, "e292_a", None, "error", with_donor=True)
        want = dict(err=lambda i: ds.table(i, "error"), clean=lambda i: ds.table(i, "exact"), gross=lambda i: ds.table(i, "gross"), none=lambda i: ds.table(i, "none"), S=lambda i: ds.table(i, "S"),
                    U=lambda i: dict(ds.table(i, "error"), count=np.zeros_like(ds.table(i, "error")["count"])), C2=lambda i: dict(ds.table(i, "error"), count=np.full_like(ds.table(i, "error")["count"], 2)))
        m.eval()
        for md in E.MODES:
            for i in (3, 7):
                b = LMD.collate([dict(LP.PTokData("val", 0, "e292_a")[i], maps=want[md](i))])
                with torch.no_grad():
                    o = m(JKT.to_device(b, "cpu"), maps=b["maps"])
                assert np.array_equal(d[md]["frames"][i]["pnone"], E.JE.batch_dump(o, b)[0]["pnone"]), f"mode {md} frame {i} is not the head on its map"
        for md in E.MODES[1:]:
            assert any(not np.array_equal(d[md]["frames"][i]["pnone"], d["err"]["frames"][i]["pnone"]) for i in range(12)), f"mode {md} = err"
        assert all(v == 0 for v in d["none"]["meta"]["use"]["n_map"]) and all(v > 0 for v in d["err"]["meta"]["use"]["n_map"]), "n_map: mode none has no samples, every real frame has a map"
        # the use report and the command: v > 0 for a block that reads the map, 12 frames with a donor; in mode S the frame's own map is the donor (v = 0)
        rep = E.use_report(d["err"])
        assert rep["frames_with_donor"] == 12 and rep["v"] > 0 and rep["v_none"] > 0 and rep["v"] != rep["v_none"] and E.use_report(d["S"])["v"] < 1e-5
        res = E.use(argparse.Namespace(run="x", split="val", max=12), runs=tmp)
        assert os.path.exists(os.path.join(tmp, "x", "map_use_val_n12.json")) and res["v_frame"] == rep["v_frame"] and len(res["v_frame"]) == 12 and res["frames"] == 12
        refuses(lambda: E.use(argparse.Namespace(run="x", split="val", max=12), runs=tmp), "never overwritten")
        with open(os.path.join(tmp, "x_none", "model.pt"), "ab") as fh:                                       # a sibling folder holds the same model.pt as the run or the dump is refused
            fh.write(b"0")
        refuses(lambda: E.dump(ns("none", mx=5), runs=tmp), "is not a copy")
        # the TRAIN sample (873 evenly spaced frames of the 8414, not the first ones): the same dump on 6 of them, the error map of the arm's loader, the scorer's reader accepts it, the use report
        from jtoken_eval import sample as jsample
        ds_tr = LP.PTokData("train", 0, "e292_a")
        want_idx = jsample(len(ds_tr), "train", 1)[:6].tolist()
        assert want_idx != list(range(6)) and want_idx[1] > 5, want_idx
        ptr = E.dump(ns("err", split="train", mx=6), runs=tmp)
        assert os.path.relpath(ptr, tmp) == "x/token_train_n6.pkl"
        dtr = pickle.load(open(ptr, "rb"))
        assert dtr["meta"]["idx"] == want_idx and dtr["meta"]["split"] == "train" and dtr["meta"]["tokens"] == [ds_tr.tokens[i] for i in want_idx] and [f["token"] for f in dtr["frames"]] == dtr["meta"]["tokens"], "the TRAIN dump is not the TRAIN sample"
        mt = dtr["meta"]                                                                                  # what the scorer's reader checks (`jtoken_eval.read_dump`; it also loads the TRAIN GT of the whole split, not done here)
        assert mt["run"] == "x" and mt["model"] == "model.pt" and mt["every"] == 1 and mt["drop"] == 0.0 and mt["drop_seed"] == 0 and mt["mode"] == "err" and len(dtr["frames"]) == 6
        dtr_ds = LMD.MapData("train", 0, "e292_a", None, "error", with_donor=True)
        for jj in (1, 4):
            b = LMD.collate([dict(ds_tr[want_idx[jj]], maps=dtr_ds.table(want_idx[jj]))])
            with torch.no_grad():
                o = m(JKT.to_device(b, "cpu"), maps=b["maps"])
            assert np.array_equal(dtr["frames"][jj]["pnone"], E.JE.batch_dump(o, b)[0]["pnone"]), f"TRAIN frame {want_idx[jj]} is not the head on its error map"
        rep_tr = E.use_report(dtr)
        assert rep_tr["frames_with_donor"] == 6 and rep_tr["v"] > 0 and rep_tr["split"] == "train"
        # params and time (the strict loader of an arm-m run, `jtoken_eval.timing`'s protocol): the block's 20 264 parameters, the head without it, GMACs with and without the block, ms per frame on the CPU, a new file every time
        import jtoken as JK
        pm = E.params(argparse.Namespace(run="x"), runs=tmp)
        assert pm["params_block"] == 20264 == JK.count_params(m.map_block) and pm["params"] == JK.count_params(m) and pm["params_head"] == pm["params"] - 20264 == JK.count_params(JKT.build_head(args)) and not pm["no_map"], pm
        tm_ = E.timing(argparse.Namespace(run="x", cpu=True, frames=3, repeats=1, K=64), runs=tmp)
        assert tm_["K"] == 64 and tm_["frames"] == 3 and tm_["n_timings"] == 3 * 1 and tm_["params"] == pm["params"] and tm_["params_block"] == 20264 and tm_["gmacs_head"] > tm_["gmacs_no_map"] > 0 and tm_["median_ms"] > 0 and tm_["p90_ms"] >= tm_["median_ms"] and tm_["map_samples_mean"] > 50 and not tm_["usable_for_gcost"] and tm_["device"] == "cpu", tm_
        assert [f for f in os.listdir(os.path.join(tmp, "x")) if f.startswith("map_timing_cpu_")], "timing wrote no file"
        refuses(lambda: E.timing(argparse.Namespace(run="x", cpu=True, frames=2, repeats=1, K=100000), runs=tmp), "is above the K of the cache")
        try:                                                                                              # `jtoken_train.load_model` builds the head without the block and cannot read an arm-m checkpoint: the reason for `load_model` here
            JKT.load_model("x", "cpu", tmp)
            raise AssertionError("the plain loader read an arm-m checkpoint")
        except RuntimeError:
            pass
        assert isinstance(E.load_model("x", "cpu", tmp)[0].map_block, type(m.map_block)) and E.load_model("x", "cpu", tmp)[1]["seed"] == 0
        # arm z: no block, no map built, every mode the same plain head on the frames, v zero
        torch.manual_seed(3)
        mz = T.build_head(dict(no_map=True))
        os.makedirs(os.path.join(tmp, "z"))
        torch.save(dict(state=mz.state_dict(), args=dict(args, no_map=True), steps=12000), os.path.join(tmp, "z", "model.pt"))
        dz = {md: pickle.load(open(E.dump(ns(md, "z", mx=6), runs=tmp), "rb")) for md in ("err", "none", "S")}
        pz = E.params(argparse.Namespace(run="z"), runs=tmp)
        tz = E.timing(argparse.Namespace(run="z", cpu=True, frames=2, repeats=1, K=64), runs=tmp)
        assert pz["params_block"] == 0 and pz["params"] == pz["params_head"] and pz["no_map"] and tz["n_timings"] == 2 and tz["params_block"] == 0 and tz["gmacs_head"] == tz["gmacs_no_map"] > 0 and tz["map_samples_mean"] == 0.0, (pz, tz)
        assert all(np.array_equal(dz["err"]["frames"][i]["pnone"], dz[md]["frames"][i]["pnone"]) for md in ("none", "S") for i in range(6)) and sum(dz["err"]["meta"]["use"]["n"]) == 0 and dz["err"]["meta"]["maps"] == "none (arm z)", "arm z: every mode is the same head"


def selftest(train_path=TRAIN, eval_path=EVAL):
    import jtoken_train as JKT
    assert JKT.build_head is not None
    T, E = load(train_path, "jmap_train_under_test"), load(eval_path, "jmap_eval_under_test")
    hook_refusal(T)
    assert has_hook(), "the real head has no S3 hook: jtoken.JTokenHead(map_block=, forward(batch, past, maps))"
    for label, stub in (("stub", make_stub()), ("real", None)):
        with Patch(stub):
            trainer_tests(T, stub, label)
            eval_tests(E, T, stub, label)
            equals_jtoken_train(T, label)
        print(f"jmap trainer + read-out tests: the {label} head ok", flush=True)
    with Patch(make_stub()):
        dump_on_real_frames(E, T)
    print("jmap_train selftest ok")


def real_loader_check(train_path=TRAIN):
    """the loader on the real cache and the lane GT store (no head): the batch of arm z is `lane_ptok_data`'s bit for bit (the same sampler order), the batch of arm m adds the maps of the card's mix, the VAL maps are the `error` maps, a donor is the `S` map, and `train()` hands the right
    arguments to `lane_map_data.loader` (the smoke path never calls it). Not part of the selftest: it needs the cache"""
    import numpy as np
    import torch
    import jtoken_train as JKT
    import lane_map_data as LMD
    import lane_ptok_data as LP
    import trainer as TR
    T = load(train_path, "jmap_train_real")
    def first(loader_fn, seed, n=2):
        TR.seed_all(seed)
        return [b for _, b in zip(range(n), loader_fn()[1])]
    ref = first(lambda: LP.loader("train", 0, 4, 0), 7)
    z = first(lambda: LMD.loader("train", 0, 4, 0, init_seed=5, maps=False), 7)
    mm = first(lambda: LMD.loader("train", 0, 4, 0, init_seed=5, maps=True), 7)
    for r_, z_, m_ in zip(ref, z, mm):
        assert sorted(z_) == sorted(r_) and sorted(set(m_) - set(r_)) == ["maps"], "the batch keys"
        for k in r_:
            if torch.is_tensor(r_[k]):
                assert same_t(r_[k], z_[k]) and same_t(r_[k], m_[k]), f"the batch of arm z / arm m differs from lane_ptok_data's at {k}"
        assert sorted(m_["maps"]) == sorted(LMD.KEYS) and m_["maps"]["pos"].shape[0] == 4 and m_["maps"]["valid"].dtype == torch.bool and m_["maps"]["valid"].shape[1] >= 1
    ds, dl = LMD.loader("val", 0, 3, 0, with_donor=True)
    b = next(iter(dl))
    assert sorted(b["maps"]) == sorted(LMD.KEYS) and sorted(b["donor"]) == sorted(LMD.KEYS)
    for j in range(3):
        for key, mode in (("maps", "error"), ("donor", "S")):
            t = ds.table(j, mode)
            n = len(t["valid"])
            assert b[key]["valid"][j].sum() == n and np.array_equal(b[key]["pos"][j, :n].numpy(), t["pos"]) and np.array_equal(b[key]["count"][j, :n].numpy(), t["count"]) and not b[key]["valid"][j, n:].any(), f"VAL {key} of frame {j}"
    do_, dlo = LMD.loader("train", 0, 4, 0, init_seed=5, mode="other")
    bo_ = next(iter(dlo))
    assert do_.mode == "other" and int(do_.draws.sum()) == 4 and sorted(bo_["maps"]) == sorted(LMD.KEYS) and bo_["maps"]["valid"].shape[0] == 4, "the loader of the capacity control"
    assert ds.mode == "error" and LMD.loader("train", 0, 2, 0)[0].mode == "train" and LMD.loader("val", 0, 2, 0, mode="exact")[0].mode == "exact"
    calls, orig = [], LMD.loader

    def recording(*a, **k):
        calls.append((a, k))
        return orig(*a, **k)
    LMD.loader = recording
    try:
        with Patch(make_stub()), tempfile.TemporaryDirectory() as tmp:
            T.train(T.parse_args(["--name", "rl", "--seed", "1", "--init-seed", "0", "--bs", "2", "--steps", "3", "--workers", "0"]), runs=tmp, ckpt_every=500)
            T.train(T.parse_args(["--name", "rz", "--seed", "1", "--bs", "3", "--steps", "2", "--workers", "0", "--no-map"]), runs=tmp, ckpt_every=500)
            T.train(T.parse_args(["--name", "ro", "--seed", "1", "--bs", "2", "--steps", "2", "--workers", "0", "--map-mode", "other"]), runs=tmp, ckpt_every=500)
            ms, args = T.load_model("rl", "cpu", tmp)
            assert ms._step_py == 3 and args["seed"] == 1 and json.load(open(os.path.join(tmp, "rl", "config.json")))["seed"] == 0 and os.path.exists(os.path.join(tmp, "rl", ".done"))
    finally:
        LMD.loader = orig
    want_calls = [(("train", 1, 2, 0), dict(init_seed=0, maps=True, mode="train", ident="e292_a", K=None)), (("train", 1, 3, 0), dict(init_seed=1, maps=False, mode="train", ident="e292_a", K=None)), (("train", 1, 2, 0), dict(init_seed=1, maps=True, mode="other", ident="e292_a", K=None))]
    assert calls == want_calls, f"the loader calls of train(): {calls}"
    print(f"real loader ok: arm z batches = lane_ptok_data's, maps M {[int(x['maps']['valid'].shape[1]) for x in mm]}")


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
        real_loader_check(sys.argv[2] if len(sys.argv) > 2 else TRAIN)
        sys.exit(0)
    for args_ in (["one", TRAIN, EVAL], ["real"]):
        r = subprocess.run([sys.executable, os.path.abspath(__file__)] + args_, capture_output=True, text=True, env=ENV)
        assert r.returncode == 0, f"{args_[0]} failed:\n{r.stdout[-1500:]}{r.stderr[-2500:]}"
        print(r.stdout.strip().splitlines()[-1])
    if len(sys.argv) < 2 or sys.argv[1] != "selftests":
        left = run_mutants()
        assert not left, f"mutants survived: {left}"
    print("test_jmap_train ok")
