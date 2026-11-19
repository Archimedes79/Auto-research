"""Tests of the E389 route-association head (`grid/jroute.py`) and its trainer (`train/jroute_train.py`); CPU only (the GPU is hidden), SYNTHETIC batches in the layout of the route loader: no cache needed.

    wsl bash tools/env/_py14.sh test_jroute.py             # the selftest of jroute.py, the trainer tests, then the textual mutants of both files: every mutant must make a test FAIL
    wsl bash tools/env/_py14.sh test_jroute.py selftests   # the selftest and the trainer tests only
    wsl bash tools/env/_py14.sh test_jroute.py mutants [jroute.py | jroute_train.py]   # the mutants only (of one file)

jroute.py selftest: pinned parameter counts (full, geometry-only), `pair_features` by hand (metre scale, clipping, both heading frames), a float64 loop reference of the head (maps, padding route keys, samples without a map, both input sets), the attention and the
pooled features by hand (A: one third each; B: the nearest key), a token 1 m and one 6 m from a route polyline, the none key = the no-map arm, the invariances (route-key order, token order, padding tokens), no GT key read, a fifth route key refused, the loss by hand
(label -1, the weights), the label frequencies, a gradient into every parameter (and none into the route side without a map), K = 512 / M = 160 memory and bf16, the data contract, and the LEARNABILITY tests (the exact-map rule > 0.95 with the route and not without it,
the straight corridor prior without a map > 0.95, the geometry-only head fits the rule). Trainer tests: options, the keys that reach the device, arm z = the route stripped (the loss is that of the explicit no-map batch and does not see the route content), the log
windows by a scripted loss, the first-batch contract check, the positive weights by hand, the smoke run with files / arms / inputs / init seed, kill + resume = the straight run, the weights of the held config reused, every recipe key refused at a resume, the loader call
(the cache seed, the mode, the same call for both arms, the first 50 batches for the weights, the data order follows the init seed), a lazy loader import. A mutant = one textual change of the source; the mutated copy runs in a temporary folder, two at a time.
"""
import concurrent.futures as cf
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import types

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HEAD, TRAIN = os.path.join(ROOT, "training", "grid", "jroute.py"), os.path.join(ROOT, "training", "train", "jroute_train.py")
ENV = dict(os.environ, BFLC_ROOT=ROOT, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")      # a shared box: one thread per test process, no GPU
for sub_ in ("training", "training/grid", "training/data", "training/data/prep", "training/train", "evaluation"):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))

H0, T0 = "jroute.py", "jroute_train.py"
MUTANTS = [
    # the pair features
    (H0, "across is the right normal", "nq, nk = (torch.stack([-t[..., 1], t[..., 0]], -1) for t in (tq, tk))", "nq, nk = (torch.stack([t[..., 1], -t[..., 0]], -1) for t in (tq, tk))"),
    (H0, "the sin of the turn has the other sign", "cross = tq[..., 0] * tk[..., 1] - tq[..., 1] * tk[..., 0]", "cross = tq[..., 1] * tk[..., 0] - tq[..., 0] * tk[..., 1]"),
    (H0, "the offset is query minus key", "ch = pos_k[:, None, :, :] - pos_q[:, :, None, :]", "ch = pos_q[:, :, None, :] - pos_k[:, None, :, :]"),
    (H0, "the distance is not scaled", "((dot(ch, ch) + 1e-12).sqrt() / METRE).clamp(max=CLIP)", "((dot(ch, ch) + 1e-12).sqrt()).clamp(max=CLIP)"),
    (H0, "the offsets are not clipped", "m = lambda v: (v / METRE).clamp(-CLIP, CLIP)  # noqa: E731", "m = lambda v: (v / METRE)  # noqa: E731"),
    (H0, "the metre scale is 20 m (the old geometry)", "HALF, METRE, CLIP, TAU = 50.0, 5.0, 3.0, 0.3", "HALF, METRE, CLIP, TAU = 50.0, 20.0, 3.0, 0.3"),
    (H0, "the clip is 30", "HALF, METRE, CLIP, TAU = 50.0, 5.0, 3.0, 0.3", "HALF, METRE, CLIP, TAU = 50.0, 5.0, 30.0, 0.3"),
    (H0, "the key-frame offsets are the token-frame ones", "m(dot(ch, tk)), m(dot(ch, nk))", "m(dot(ch, tq)), m(dot(ch, nq))"),
    (H0, "the lane count is not a feature", "(count_k / (NCOUNT - 1))[:, None, :].expand_as(cross)", "torch.zeros_like(cross)"),
    (H0, "the lane count is not scaled", "(count_k / (NCOUNT - 1))[:, None, :].expand_as(cross)", "count_k[:, None, :].expand_as(cross)"),
    # the attention A
    (H0, "the none key is missing", "z_none[..., None]], -1).softmax(-1)", "z_none[..., None] * 0 + NEG], -1).softmax(-1)"),
    (H0, "padding route keys are not masked", "torch.cat([z.masked_fill(~rvalid[:, None, None, :], NEG), z_none[..., None]], -1)", "torch.cat([z, z_none[..., None]], -1)"),
    (H0, "the none bias is dropped", "/ math.sqrt(self.dh) + self.none_b[None, :, None]", "/ math.sqrt(self.dh)"),
    (H0, "the attention is not scaled", "(q @ k.transpose(-1, -2)).float() / math.sqrt(self.dh) + self.bias(g)", "(q @ k.transpose(-1, -2)).float() + self.bias(g)"),
    (H0, "the geometry bias is missing", "/ math.sqrt(self.dh) + self.bias(g).float().permute(0, 3, 1, 2)", "/ math.sqrt(self.dh)"),
    (H0, "the none value is left out of the output", "(att[..., :-1].to(v.dtype) @ v + att[..., -1:].to(v.dtype) * vn)", "(att[..., :-1].to(v.dtype) @ v)"),
    (H0, "the route key has no lane count", 'F.one_hot(rt["count"].long(), NCOUNT).float()', "torch.zeros(B, M, NCOUNT)"),
    (H0, "the route position is not scaled in the key", "rf = torch.cat([rpos / HALF, rdir,", "rf = torch.cat([rpos, rdir,"),
    (H0, "the mass is the none mass", 'mass=att[..., :-1].sum(-1).transpose(1, 2)', 'mass=att[..., -1].transpose(1, 2)'),
    (H0, "the pooled features are not weighted by A", "pm_a=(att[..., :-1].mean(1)[..., None] * g.float()).sum(2)", "pm_a=g.float().mean(2)"),
    # the sharp attention B
    (H0, "attention B takes the farthest key", "(-g[..., 6] * (METRE / TAU))", "(g[..., 6] * (METRE / TAU))"),
    (H0, "attention B has a high temperature", "HALF, METRE, CLIP, TAU = 50.0, 5.0, 3.0, 0.3", "HALF, METRE, CLIP, TAU = 50.0, 5.0, 3.0, 3.0"),
    (H0, "attention B does not mask padding keys", ".masked_fill(~rvalid[:, None, :], NEG).softmax(-1) * anyv", ".softmax(-1) * anyv"),
    (H0, "attention B is not zeroed without a map", ".masked_fill(~rvalid[:, None, :], NEG).softmax(-1) * anyv", ".masked_fill(~rvalid[:, None, :], NEG).softmax(-1)"),
    (H0, "A and B are swapped in the output", 'for n in ("x", "attended", "mass", "pm_a", "pm_b", "anyv")', 'for n in ("x", "attended", "mass", "pm_b", "pm_a", "anyv")'),
    # the tokens
    (H0, "the own position has no metre-scale feature", "torch.cat([pos / HALF, (pos / METRE).clamp(-CLIP, CLIP), dirv,", "torch.cat([pos / HALF, pos / HALF, dirv,"),
    (H0, "p_exist is the logit", "e.sigmoid()[..., None], flags.sigmoid()], -1)", "e[..., None], flags.sigmoid()], -1)"),
    (H0, "the flags are logits", "e.sigmoid()[..., None], flags.sigmoid()], -1)", "e.sigmoid()[..., None], flags], -1)"),
    (H0, "the forward reads a label", "b, rt = {k: batch[k] for k in INPUT_SETS[self.inputs]}, batch[\"route\"]", "b, rt = {k: batch[k] for k in INPUT_SETS[self.inputs] + (\"on_road\",)}, batch[\"route\"]"),
    (H0, "the geometry-only head reads tok", 'geo=("pos", "dirv", "e", "flags"))', 'geo=("tok", "pos", "dirv", "e", "flags"))'),
    (H0, "a fifth route key is accepted", "assert set(route) == set(ROUTE_KEYS),", "assert set(ROUTE_KEYS) <= set(route),"),
    (H0, "the head is narrower", 'def __init__(self, d=D, heads=4, geo_hidden=32, hidden=64, inputs="full"):', 'def __init__(self, d=D, heads=4, geo_hidden=32, hidden=48, inputs="full"):'),
    # the loss, the frequencies, the contract
    (H0, "the label -1 is not ignored", "m = y >= 0", "m = y >= -1"),
    (H0, "the weights are swapped", 'zip(("road", "lane"), LABELS, pos_weight)', 'zip(("road", "lane"), LABELS, pos_weight[::-1])'),
    (H0, "the weight is on the negatives", "torch.where(y == 1, float(w), 1.0)", "torch.where(y == 0, float(w), 1.0)"),
    (H0, "the loss is a sum", "term = (bce * m).sum() / n.clamp_min(1)", "term = (bce * m).sum()"),
    (H0, "the weight is n_pos / n_neg", "float(pn[n][1] / pn[n][0])", "float(pn[n][0] / pn[n][1])"),
    (H0, "the counts are swapped", "{n: (int((batch[k] == 1).sum()), int((batch[k] == 0).sum()))", "{n: (int((batch[k] == 0).sum()), int((batch[k] == 1).sum()))"),
    (H0, "a lane token off the road is accepted", 'assert bool((batch["on_road"][batch["on_lane"] == 1] == 1).all()), "a token on the ego lane is not on the road"', "pass"),
    (H0, "a padding token may have a label", 'assert bool((y[~batch["valid"].bool()] == -1).all()), f"{key}: a padding token has a label"', "pass"),
    (H0, "the lane count range is not checked", 'assert int(rt["count"].min()) >= 0 and int(rt["count"].max()) <= NCOUNT - 1, "the lane count is outside 0 - 5"', "pass"),
    # the trainer
    (T0, "arm z does not strip the route", '        if arm == "z":\n            b = strip_route(b)', "        if False:\n            b = strip_route(b)"),
    (T0, "the arms are swapped", '        if arm == "z":\n            b = strip_route(b)', '        if arm == "m":\n            b = strip_route(b)'),
    (T0, "the strip keeps the valid flags", 'route=dict(b["route"], valid=torch.zeros_like(b["route"]["valid"]))', 'route=dict(b["route"], valid=b["route"]["valid"])'),
    (T0, "the positive weights are swapped", "JR.route_loss(m(b), b, pos_weight)", "JR.route_loss(m(b), b, pos_weight[::-1])"),
    (T0, "another recipe is not refused", 'assert c.get("design") == "jroute" and old == recipe_of(args)', 'assert c.get("design") == "jroute" or old == recipe_of(args)'),
    (T0, "the arm is not part of the recipe", 'RECIPE = ("seed", "init_seed", "cache", "ident", "K", "bs", "steps", "lr", "arm", "inputs")', 'RECIPE = ("seed", "init_seed", "cache", "ident", "K", "bs", "steps", "lr", "inputs")'),
    (T0, "the inputs are not part of the recipe", 'RECIPE = ("seed", "init_seed", "cache", "ident", "K", "bs", "steps", "lr", "arm", "inputs")', 'RECIPE = ("seed", "init_seed", "cache", "ident", "K", "bs", "steps", "lr", "arm")'),
    (T0, "the init seed is ignored", "init = a.seed if a.init_seed is None else a.init_seed", "init = a.seed"),
    (T0, "the loop is seeded by the cache seed", "seed=init, args=args,", "seed=a.seed, args=args,"),
    (T0, "the weights are re-estimated at a resume", 'pw = held["pos_weight"] if held and held.get("pos_weight") else estimate_pos_weight(first)', "pw = estimate_pos_weight(first)"),
    (T0, "the weights use 5 batches", "itertools.islice(loader, FREQ_BATCHES)", "itertools.islice(loader, 5)"),
    (T0, "everything of the batch goes to the device", "{k: move(b[k], dev) for k in JR.needed(inputs)}", "{k: move(v, dev) for k, v in b.items() if not isinstance(v, list)}"),
    (T0, "the log window is 10", "WINDOW = 100 ", "WINDOW = 10 "),
    (T0, "the first batch is not checked", "        if n[0] == 1:\n            JR.check_batch(b)\n", "        if n[0] == 99:\n            JR.check_batch(b)\n"),
    (T0, "the route cache identity is not stored", 'args["route_cache"] = dict(split="train"', 'args["_route_cache"] = dict(split="train"'),
    (T0, "a resume on another route cache is accepted", 'assert not held or held.get("route_cache") in (None, args["route_cache"])', 'assert True or held.get("route_cache") in (None, args["route_cache"])'),
    (T0, "the route cap is not in the identity", 'max_samples=man["params"]["max_samples"], tokens_sha', 'max_samples=0, tokens_sha'),
    (T0, "the loader gets the head seed", 'LRD.loader("train", a.seed, a.bs', 'LRD.loader("train", init, a.bs'),
    (T0, "the loader gets another mode", 'init_seed=init, mode="train", ident=a.ident', 'init_seed=init, mode="error", ident=a.ident'),
    (T0, "the loader gets the cache seed as init seed", "a.workers, init_seed=init, mode", "a.workers, init_seed=a.seed, mode"),
    (T0, "the loader gets no init seed", "a.workers, init_seed=init, mode", "a.workers, mode"),
    (T0, "the smoke run has 25 steps", "steps=20 if a.smoke else a.steps", "steps=25 if a.smoke else a.steps"),
    (T0, "the labelled share counts the padding", 'nv = max(float(b["valid"].sum()), 1.0)', 'nv = max(float(b["valid"].numel()), 1.0)'),
    (T0, "the geometry-only head is the full head", 'JR.RouteHead(inputs=args.get("inputs") or "full")', "JR.RouteHead()"),
    (T0, "an accuracy step without a label enters the window", '            if info[f"n_{k}"] > 0:\n                win[f"acc_{k}"]', '            if True:\n                win[f"acc_{k}"]'),
    (T0, "the positive share is of all tokens", 'win[f"pos_{k}"].append(info[f"pos_{k}"] / info[f"n_{k}"])', 'win[f"pos_{k}"].append(info[f"pos_{k}"] / nv)'),
]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def refuses(fn, text):
    """fn must fail with an AssertionError (or a SystemExit for argparse) whose message holds `text`"""
    try:
        fn()
    except AssertionError as e:
        assert text in str(e), (text, str(e))
        return
    raise RuntimeError(f"accepted although it must be refused ({text})")


# ------------------------------------------------------------------------------------------------------------------------------ the trainer
def trainer_tests(T):
    import numpy as np
    import torch
    JR = T.JR
    B, K, M = 4, 24, 12
    b0 = JR.synthetic_batch(B, K, M, seed=3)
    # options
    a = T.parse_args(["--name", "x"])
    assert (a.arm, a.inputs, a.steps, a.bs, a.lr, a.seed, a.init_seed, a.ident) == ("m", "full", 3000, 8, 1e-3, 0, None, "e292_a"), a
    z = T.parse_args(["--name", "x", "--arm", "z", "--inputs", "geo"])
    assert z.arm == "z" and z.inputs == "geo"
    for bad in (["--arm", "q"], ["--inputs", "tok"]):
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                T.parse_args(["--name", "x"] + bad)
                raise RuntimeError(f"{bad} accepted")
            except SystemExit:
                pass
    refuses(lambda: T.parse_args(["--name", "x", "--bs", "0"]), "--bs 0")
    refuses(lambda: T.main(["--name", "x"], runs="nowhere"), "no GPU")
    # only the keys the head and the loss read reach the device: no dense GT target, no label dict, no list; the route is moved key by key; geo has no tok
    host = dict(b0, y=torch.zeros(B, K, K), ytype=torch.zeros(B, K, dtype=torch.long), label=dict(gt_id=torch.zeros(B, K)), tokens_np=[{}] * B, token=["a"] * B, e_raw=torch.zeros(B, K), s=torch.zeros(B, K))
    for inputs in ("full", "geo"):
        d = T.to_device(host, inputs, "cpu")
        assert set(d) == set(JR.needed(inputs)) and set(d["route"]) == set(JR.ROUTE_KEYS) and d["route"]["count"].dtype == torch.int64 and d["on_road"].dtype == torch.int8, (inputs, sorted(d))
    assert "tok" in T.to_device(host, "full", "cpu") and "tok" not in T.to_device(host, "geo", "cpu")
    # arm z = the route stripped: the loss is that of the explicit no-map batch, is blind to the route content; arm m reads it; the weights are (road, lane)
    head = JR._trained(5)
    pw = (3.0, 5.0)
    junk = dict(b0, route=dict(b0["route"], pos=b0["route"]["pos"] + 31.0, dir=b0["route"]["dir"].flip(-1)))
    ref_m = JR.route_loss(head(b0), b0, pw)
    ref_z = JR.route_loss(head(JR._no_map(b0)), b0, pw)
    lm, lz = T.make_loss("m", "full", pw), T.make_loss("z", "full", pw)
    lm_, im_ = lm(head, b0)
    lz_, iz_ = lz(head, b0)
    assert abs(float(lm_) - float(ref_m[0])) < 1e-6 and abs(im_["bce_road"] - ref_m[1]["bce_road"]) < 1e-6 and abs(im_["bce_lane"] - ref_m[1]["bce_lane"]) < 1e-6, "arm m is the plain loss"
    assert abs(float(lz_) - float(ref_z[0])) < 1e-6 and abs(float(lz_) - float(ref_m[0])) > 1e-4, "arm z is the loss of the no-map batch"
    assert abs(float(T.make_loss("z", "full", pw)(head, junk)[0]) - float(lz_)) < 1e-6, "arm z saw the content of the route"
    assert abs(float(T.make_loss("m", "full", pw)(head, junk)[0]) - float(lm_)) > 1e-4, "arm m does not read the route"
    assert abs(float(T.make_loss("m", "full", (1.0, 1.0))(head, b0)[0]) - float(lm_)) > 1e-3, "the positive weights are not used"
    sb = T.strip_route(b0)
    assert not bool(sb["route"]["valid"].any()) and bool(b0["route"]["valid"].any()) and sb["route"]["pos"] is b0["route"]["pos"] and sb["pos"] is b0["pos"], "strip_route"
    # the first batch is checked against the data contract (a padding token with a label), the later ones are not (a sync per step)
    bad = dict(b0, on_road=torch.ones_like(b0["on_road"]), on_lane=torch.ones_like(b0["on_lane"]))
    refuses(lambda: T.make_loss("m", "full", pw)(head, bad), "padding token")
    T.make_loss("m", "full", pw)(head, b0)
    # the log windows by a scripted loss: acc / lab / pos are the means of the last WINDOW steps; a step without a labelled token is not in the accuracy and the positive share, it is in the labelled share (0)
    nvalid = float(b0["valid"].sum())
    calls = []

    def fake(out, batch, pos_weight):
        t = len(calls) + 1
        n = 0.0 if t == 5 else 10.0
        calls.append(t)
        return out["road"].sum() * 0.0, dict(bce_road=float(t), bce_lane=2.0 * t, acc_road=t / 200.0, acc_lane=t / 400.0, n_road=n, n_lane=n / 2, pos_road=n / 5, pos_lane=n / 10)
    real = JR.route_loss
    JR.route_loss = fake
    try:
        lw = T.make_loss("m", "full", pw)
        for _ in range(150):
            _, iw = lw(head, b0)
    finally:
        JR.route_loss = real
    ts = [t for t in range(1, 151)]
    last = ts[-T.WINDOW:]
    assert iw["bce_road"] == 150.0 and iw["bce_lane"] == 300.0, iw
    assert abs(iw["acc_road"] - np.mean([t / 200.0 for t in last])) < 1e-9 and abs(iw["acc_lane"] - np.mean([t / 400.0 for t in last])) < 1e-9, ("acc window", iw)
    assert abs(iw["lab_road"] - np.mean([(0.0 if t == 5 else 10.0) / nvalid for t in last])) < 1e-9 and abs(iw["lab_lane"] - np.mean([(0.0 if t == 5 else 5.0) / nvalid for t in last])) < 1e-9, ("lab window", iw)
    assert abs(iw["pos_road"] - 0.2) < 1e-9 and abs(iw["pos_lane"] - 0.2) < 1e-9, ("positive share of the labelled tokens", iw)
    assert T.WINDOW == 100
    JR.route_loss = fake
    calls.clear()
    try:
        lw = T.make_loss("m", "full", pw)
        for _ in range(6):
            _, i6 = lw(head, b0)
    finally:
        JR.route_loss = real
    assert abs(i6["acc_road"] - np.mean([t / 200.0 for t in (1, 2, 3, 4, 6)])) < 1e-9 and abs(i6["lab_road"] - np.mean([(0.0 if t == 5 else 10.0) / nvalid for t in range(1, 7)])) < 1e-9, ("a step without a label", i6)
    # the positive weights by hand: n_neg / n_pos over the batches
    t1 = dict(on_road=torch.tensor([[1, 0, 0, -1]], dtype=torch.int8), on_lane=torch.tensor([[1, 1, 0, 0]], dtype=torch.int8))
    t2 = dict(on_road=torch.tensor([[1, 1, 0, 0]], dtype=torch.int8), on_lane=torch.tensor([[0, 0, 0, -1]], dtype=torch.int8))
    assert T.estimate_pos_weight([t1, t2]) == (4 / 3, 5 / 2), T.estimate_pos_weight([t1, t2])        # road: 3 positives, 4 negatives; lane: 2 positives, 5 negatives
    # the smoke run: files, the counter, the arms, the inputs, the init seed, the weights in the args
    with tempfile.TemporaryDirectory() as tmp:
        sys.modules["lane_route_data"] = None                                                                  # the smoke path never imports the loader
        try:
            def smoke(*argv, **kw):
                T.train(T.parse_args(["--name", argv[0], "--smoke", "--bs", "2"] + list(argv[1:])), runs=tmp, ckpt_every=10, smoke_k=24, **kw)
            smoke("r", "--seed", "0")
            for f in (".done", "model.pt", "config.json", "log.jsonl", "ckpt.pt"):
                assert os.path.exists(os.path.join(tmp, "r", f)), f"{f} missing"
            m, args = T.load_model("r", "cpu", tmp)
            cfg = json.load(open(os.path.join(tmp, "r", "config.json")))
            assert cfg["design"] == "jroute" and cfg["steps"] == 20 and args["arm"] == "m" and args["inputs"] == "full" and args["init_seed"] == 0 and len(args["pos_weight"]) == 2 and all(w > 0 for w in args["pos_weight"]), (cfg, args)
            rows = [json.loads(line) for line in open(os.path.join(tmp, "r", "log.jsonl"))]
            assert rows[-1]["step"] == 20 and {"bce_road", "bce_lane", "acc_road", "acc_lane", "lab_road", "lab_lane", "pos_road", "pos_lane"} <= set(rows[-1]) and all(np.isfinite(r["total"]) for r in rows), rows[-1]
            torch.manual_seed(0)
            assert not torch.equal(m.out[2].weight, JR.RouteHead().out[2].weight), "the head did not train"
            smoke("r")                                                                                           # the same recipe in a finished folder is resumed
            smoke("zz", "--arm", "z")
            mz, az = T.load_model("zz", "cpu", tmp)
            assert az["arm"] == "z" and isinstance(mz, JR.RouteHead) and mz.inputs == "full"
            smoke("gg", "--inputs", "geo")
            mg, ag = T.load_model("gg", "cpu", tmp)
            assert ag["inputs"] == "geo" and mg.inputs == "geo" and not any(n.startswith("tok") for n, _ in mg.named_parameters()) and sum(p.numel() for p in mg.parameters()) == 21130
            smoke("i", "--seed", "0", "--init-seed", "3")
            mi, ai = T.load_model("i", "cpu", tmp)
            assert json.load(open(os.path.join(tmp, "i", "config.json")))["seed"] == 3 and ai["seed"] == 0 and ai["init_seed"] == 3 and cfg["seed"] == 0, "--init-seed is the seed of the loop"
            assert not all(torch.equal(p, q) for p, q in zip(m.state_dict().values(), mi.state_dict().values())), "another init seed gives the same head"
            smoke("r2", "--seed", "0")
            m2, _ = T.load_model("r2", "cpu", tmp)
            assert all(torch.equal(p, q) for p, q in zip(m.state_dict().values(), m2.state_dict().values())), "the same recipe is not reproducible"
            # every recipe key is refused at a resume
            folder = os.path.join(tmp, "r")
            change = dict(seed=lambda v: v + 1, init_seed=lambda v: v + 1, cache=lambda v: "another_root", ident=lambda v: "e273_p", K=lambda v: 384, bs=lambda v: v + 1, steps=lambda v: v + 1, lr=lambda v: v * 2,
                          arm=lambda v: "z" if v == "m" else "m", inputs=lambda v: "geo" if v == "full" else "full")
            assert set(change) == set(T.RECIPE), f"a recipe key without a refusal test: {set(T.RECIPE) ^ set(change)}"
            assert T.check_start(args, folder)["pos_weight"] == args["pos_weight"], "a held run gives back its args"
            assert T.check_start(args, os.path.join(tmp, "nothing")) is None
            for k_ in change:
                refuses(lambda: T.check_start(dict(args, **{k_: change[k_](args[k_])}), folder), "of another run")
            refuses(lambda: smoke("r", "--seed", "1"), "of another run")
            # kill at step 13, resume: the run finishes as the straight run; a second killed run whose stored weights are edited resumes with the stored ones
            real_make, state = T.make_loss, dict(boom=True)

            def flaky(arm, inputs, pos_weight):
                f, n = real_make(arm, inputs, pos_weight), [0]

                def g(mm, bb):
                    n[0] += 1
                    if state["boom"] and n[0] == 13:
                        raise RuntimeError("killed")
                    return f(mm, bb)
                return g

            def killed_then_resumed(name, edit=None):
                state["boom"] = True
                try:
                    smoke(name, "--seed", "0")
                    raise AssertionError("the run was not killed")
                except RuntimeError as e:
                    assert "killed" in str(e)
                if edit:
                    cp = os.path.join(tmp, name, "config.json")
                    c_ = json.load(open(cp))
                    c_["args"]["pos_weight"] = edit
                    json.dump(c_, open(cp, "w"))
                state["boom"] = False
                smoke(name, "--seed", "0")
                return T.load_model(name, "cpu", tmp)
            T.make_loss = flaky
            try:
                mk, ak = killed_then_resumed("k")
                _, ae = killed_then_resumed("ke", [7.0, 9.0])
            finally:
                T.make_loss = real_make
            assert ak["pos_weight"] == args["pos_weight"] and ae["pos_weight"] == [7.0, 9.0], "a resume re-estimated the weights"
            assert all(torch.allclose(p, q, atol=1e-4) for p, q in zip(m.state_dict().values(), mk.state_dict().values()) if p.dtype.is_floating_point), "the resumed run is not the straight run"
        finally:
            sys.modules.pop("lane_route_data", None)
    # the loader call and the data order through a fake loader module: ("train", the CACHE seed, bs, workers) and mode / ident / K / cache, the same call for both arms, the weights from the first 50 batches, the order follows the init seed
    nb = 60
    ds = []
    for s in range(nb):
        bt = JR.synthetic_batch(2, 16, 6, seed=500 + s)
        if s >= 50:                                                                                           # the batches after the 50th have another label ratio: they must not enter the weights
            bt["on_road"] = bt["on_lane"] = torch.where(bt["valid"], 1, -1).to(torch.int8)
        ds.append(dict(bt, idx=torch.tensor(s)))
    want_pw = T.estimate_pos_weight(ds[:50])
    seen, order = [], []
    fake = types.ModuleType("lane_route_data")

    def loader(split, seed, bs, workers, init_seed=None, mode=None, ident=None, K=None, cache=None):
        seen.append((split, seed, bs, workers, init_seed, mode, ident, K, cache))
        return types.SimpleNamespace(rc=types.SimpleNamespace(manifest=dict(fake.manifest))), torch.utils.data.DataLoader(ds, batch_size=None, shuffle=getattr(fake, "shuffle", False))
    fake.loader = loader
    fake.manifest = dict(version="v4", lane_gt="g1", tokens_sha="sha", params=dict(max_samples=160))
    sys.modules["lane_route_data"] = fake
    real_make = T.make_loss

    def recording(arm, inputs, pos_weight):
        f = real_make(arm, inputs, pos_weight)

        def g(mm, bb):
            order.append(int(bb["idx"]))
            return f(mm, bb)
        return g
    T.make_loss = recording
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for arm in ("m", "z"):
                fake.shuffle = False
                T.train(T.parse_args(["--name", arm, "--arm", arm, "--seed", "1", "--init-seed", "0", "--bs", "2", "--steps", "3", "--workers", "0", "--ident", "e273_p", "--K", "384", "--cache", "rr"]), runs=tmp)
                assert os.path.exists(os.path.join(tmp, arm, ".done")) and json.load(open(os.path.join(tmp, arm, "config.json")))["args"]["pos_weight"] == [float(w) for w in want_pw], f"arm {arm}: the weights are those of the first 50 batches"
                assert json.load(open(os.path.join(tmp, arm, "config.json")))["args"]["route_cache"] == dict(split="train", version="v4", lane_gt="g1", max_samples=160, tokens_sha="sha"), f"arm {arm}: the identity of the TRAIN route cache is stored"
            fake.manifest = dict(fake.manifest, params=dict(max_samples=64))                                       # a resume of the run on a cache with another route cap is refused
            n_seen = len(seen)
            try:
                T.train(T.parse_args(["--name", "m", "--arm", "m", "--seed", "1", "--init-seed", "0", "--bs", "2", "--steps", "3", "--workers", "0", "--ident", "e273_p", "--K", "384", "--cache", "rr"]), runs=tmp)
                raise SystemExit("a resume on another route cache was accepted")
            except AssertionError as e:
                assert "route cache" in str(e), e
            fake.manifest = dict(version="v4", lane_gt="g1", tokens_sha="sha", params=dict(max_samples=160))
            del seen[n_seen:]                                                                                      # the refused call's loader entry is not one of the loader-call assertions below
            T.train(T.parse_args(["--name", "nc", "--steps", "2", "--workers", "0"]), runs=tmp)
            assert seen[:2] == [("train", 1, 2, 0, 0, "train", "e273_p", 384, "rr")] * 2 and seen[2] == ("train", 0, 8, 0, 0, "train", "e292_a", None, None), seen
            fake.shuffle = True
            runs_ = {}
            for name, sd, ini in (("a", 0, 3), ("b", 3, 3), ("c", 0, 0)):
                order.clear()
                T.train(T.parse_args(["--name", name, "--seed", str(sd), "--init-seed", str(ini), "--bs", "2", "--steps", "6", "--workers", "0"]), runs=tmp)
                runs_[name] = list(order)
            assert runs_["a"] == runs_["b"] and runs_["a"] != runs_["c"] and len(runs_["a"]) == 6, f"the data order follows the init seed: {runs_}"
    finally:
        T.make_loss = real_make
        sys.modules.pop("lane_route_data", None)


def selftest(train_path=TRAIN):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    T = load(train_path, "jroute_train_under_test")
    trainer_tests(T)
    print("jroute_train tests ok")


# ------------------------------------------------------------------------------------------------------------------------------ mutants
def one(args):
    k, (mod, name, a, b), tmp = args
    path = HEAD if mod == H0 else TRAIN
    src = open(path, encoding="utf-8").read()
    if src.count(a) != 1:
        return name, f"NOT APPLIED ({src.count(a)} matches)"
    d = os.path.join(tmp, str(k))
    os.makedirs(d)
    p = os.path.join(d, mod)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(src.replace(a, b))
    cmd = [sys.executable, p, "selftest"] if mod == H0 else [sys.executable, os.path.abspath(__file__), "one", p]
    r = subprocess.run(cmd, capture_output=True, text=True, env=ENV)
    return name, "killed" if r.returncode != 0 else "SURVIVED"


def run_mutants(workers=2, only=None):
    """the mutants of one file (`only` = "jroute.py" | "jroute_train.py") or all"""
    ms = [m for m in MUTANTS if only in (None, m[0])]
    left = []
    with tempfile.TemporaryDirectory() as tmp, cf.ThreadPoolExecutor(workers) as ex:
        for name, res in ex.map(one, [(k, m, tmp) for k, m in enumerate(ms)]):
            print(f"{res:<12} {name}", flush=True)
            if res != "killed":
                left.append(name)
    print(f"{len(ms) - len(left)} of {len(ms)} killed", flush=True)
    return left


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        selftest(sys.argv[2])
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "mutants":                                                      # only the mutants (of one file), after a change of that file
        left = run_mutants(only=sys.argv[2] if len(sys.argv) > 2 else None)
        assert not left, f"mutants survived: {left}"
        sys.exit(0)
    for cmd in ([HEAD, "selftest"], [os.path.abspath(__file__), "one", TRAIN]):
        r = subprocess.run([sys.executable] + cmd, capture_output=True, text=True, env=ENV)
        assert r.returncode == 0, f"{os.path.basename(cmd[0])} failed:\n{r.stdout[-1500:]}{r.stderr[-2500:]}"
        print(r.stdout.strip().splitlines()[-1], flush=True)
    if len(sys.argv) < 2 or sys.argv[1] != "selftests":
        left = run_mutants()
        assert not left, f"mutants survived: {left}"
    print("test_jroute ok")
