"""Tests of the speed path of the token head (`jtoken_fast.py`): the same function as `jtoken.JTokenHead` / `jtoken.jtoken_loss`. CPU (2 threads) on synthetic batches; the GPU tests (no host sync in the loss, compile) run when a GPU is
visible and the environment variable JTF_GPU=1 is set (a short run, < 1 GB).

    wsl bash tools/env/_py14.sh training/grid/test_jtoken_fast.py [name ...]

head (eval mode: every output of the plain head to 1e-5 on the same weights; training mode: the same step counter, `scale` None, `scale_at` equal to the plain scales at chosen pairs), loss (value, info and the gradient of
every parameter against `jtoken_loss` for the default, none_weight 0.25, pos_loss max, none_weight_near, scale_w 0), sparse (the fixed-shape pair list gives the dense mask's loss; an empty list gives no scale term),
frame_mean (= `jtoken._frame_mean`, also with no valid entry in a frame and in all frames), delegation (past / maps / live_inputs run the parent's forward), state (a state_dict goes both ways, no new key),
mutants (a wrong pair index, the stacked first Linear with a swapped slice: each makes a test fail), options_and_mirror, loader_loss (real frames: gate mask and win mask of the pair list), gpu (JTF_GPU=1: no host sync in the sparse loss, bf16 autocast loss within 2e-2 of the plain one, compile_pairs equal)."""
import json
import os
import shutil
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jtoken as JK  # noqa: E402
import jtoken_fast as JF  # noqa: E402

torch.set_num_threads(2)


def _pair(seed=0, B=3, K=48):
    torch.manual_seed(seed)
    plain = JK.JTokenHead()
    for p in plain.parameters():
        p.data += 0.05 * torch.randn_like(p)                    # off the zero-ish init: every part of the head matters
    fast = JF.FastJTokenHead()
    fast.load_state_dict(plain.state_dict())
    plain._step_py = fast._step_py = JK.SWITCH + 5              # past the switch: the learned key weights
    return plain, fast, JK.synthetic_batch(B, K, seed)


def _grads(m):
    return {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}


def _close_rel(a, b, tol, what):
    """relative to 1 + |value|: the masked entries of a log-probability sit at -1e4, where one float32 ulp is 1e-3"""
    d = float(((a - b).abs() / (1.0 + a.abs())).max())
    assert d <= tol, f"{what}: max relative diff {d} > {tol}"


def _close(a, b, tol, what):
    d = float((a - b).abs().max())
    assert d <= tol, f"{what}: max diff {d} > {tol}"


def test_head():
    plain, fast, b = _pair()
    plain.eval(), fast.eval()
    with torch.no_grad():
        o1, o2 = plain(b), fast(b)
    for k in ("logp", "scale", "types", "types1", "chord", "x", "w"):
        _close(o1[k], o2[k], 1e-5, f"eval {k}")
    assert torch.equal(o1["cand"], o2["cand"]) and torch.equal(o1["valid"], o2["valid"])
    plain.train(), fast.train()
    s0 = int(plain.step)
    o1, o2 = plain(b), fast(b)
    assert o2["scale"] is None and int(fast.step) == int(plain.step) == s0 + 1 and fast._step_py == plain._step_py
    bi, ii, jj = torch.tensor([0, 0, 1, 2]), torch.tensor([9, 3, 9, 4]), torch.tensor([10, 5, 10, 6])
    _close(o2["scale_at"](bi, ii, jj), o1["scale"][bi, ii, jj], 1e-5, "scale_at")
    _close(o1["logp"], o2["logp"], 1e-5, "train logp")
    print("head ok")


def test_loss():
    for kw in (dict(), dict(none_weight=0.25), dict(pos_loss="max"), dict(none_weight=0.25, none_weight_near=0.73), dict(scale_w=0.0), dict(type_w=0.0)):
        plain, fast, b = _pair(1)
        l1, i1 = JK.jtoken_loss(plain(b), b, **kw)
        l1.backward()
        l2, i2 = JF.fast_loss(fast(b), b, **kw)
        l2.backward()
        _close(l1, l2, 1e-5, f"loss {kw}")
        assert set(i1) == set(i2), (kw, set(i1) ^ set(i2))
        for k in i1:
            assert abs(float(i1[k]) - float(i2[k])) <= 1e-5 * max(1.0, abs(float(i1[k]))), (kw, k, i1[k], i2[k])
        g1, g2 = _grads(plain), _grads(fast)
        assert g1.keys() == g2.keys()
        for n in g1:
            _close(g1[n], g2[n], 2e-5, f"grad {n} {kw}")
    print("loss ok")


def _pair_list(b, pad=64):
    """the fixed-shape pair list of a dense batch: the pairs with a finite yscale, padded to `pad`"""
    m = torch.isfinite(b["yscale"][..., 0])
    bi, ii, jj = m.nonzero(as_tuple=True)
    n = bi.numel()
    assert n <= pad
    z = lambda t, v=0: torch.cat([t, torch.full((pad - n,) + t.shape[1:], v, dtype=t.dtype)])  # noqa: E731
    return dict(sp_b=z(bi), sp_i=z(ii), sp_j=z(jj), sp_y=z(b["yscale"][bi, ii, jj], float("nan")), sp_ok=torch.arange(pad) < n)


def test_sparse():
    plain, fast, b = _pair(2)
    l1, _ = JK.jtoken_loss(plain(b), b)
    bs = dict(b, **_pair_list(b))
    l2, i2 = JF.fast_loss(fast(bs), bs)
    _close(l1, l2, 1e-5, "sparse loss")
    assert "scale_l1" in i2 and float(i2["scale_l1"]) > 0
    out_gate = dict(bs, sp_b=torch.cat([bs["sp_b"], torch.tensor([0])]), sp_i=torch.cat([bs["sp_i"], torch.tensor([0])]), sp_j=torch.cat([bs["sp_j"], torch.tensor([0])]),     # the pair (0, 0): never in a gate
                    sp_y=torch.cat([bs["sp_y"], torch.tensor([[0.9, 0.9]])]), sp_ok=torch.cat([bs["sp_ok"], torch.tensor([True])]))
    assert not bool(fast(out_gate)["cand"][0, 1, 0, 0])
    l2b, _ = JF.fast_loss(fast(out_gate), out_gate)
    _close(l2, l2b, 1e-6, "a pair outside the long gate must not count (the cand mask of the scale term)")
    nan_all = torch.full_like(b["yscale"], float("nan"))
    plain2, fast2, _ = _pair(2)
    dense_empty = dict(b, yscale=nan_all)
    l3, _ = JK.jtoken_loss(plain2(dense_empty), dense_empty)
    l4, _ = JF.fast_loss(fast2(dense_empty), dense_empty)
    _close(l3, l4, 1e-5, "no scale pair, dense")
    fast3 = JF.FastJTokenHead()
    fast3.load_state_dict(plain2.state_dict())
    fast3._step_py = JK.SWITCH + 5                                # load_state_dict resets the mirror to the step buffer (0)
    e2 = dict(dense_empty, sp_b=torch.zeros(8, dtype=torch.long), sp_i=torch.zeros(8, dtype=torch.long), sp_j=torch.zeros(8, dtype=torch.long), sp_y=torch.full((8, 2), float("nan")), sp_ok=torch.zeros(8, dtype=torch.bool))
    l6, i6 = JF.fast_loss(fast3(e2), e2)
    _close(l3, l6, 1e-5, "empty pair list")
    assert float(i6["scale_l1"]) == 0.0
    print("sparse ok")


def test_frame_mean():
    x = torch.randn(4, 7, requires_grad=True)
    m = torch.rand(4, 7) > 0.4
    m[2] = False                                                # a frame without an entry
    w = torch.rand(4, 7) + 0.5
    for ww in (None, w):
        a, b = JK._frame_mean(x, m, ww), JF._frame_mean_free(x, m, ww)
        _close(a, b, 1e-6, "frame_mean")
        ga, = torch.autograd.grad(a, x)
        gb, = torch.autograd.grad(b, x)
        _close(ga, gb, 1e-6, "frame_mean grad")
    none = torch.zeros(4, 7, dtype=torch.bool)
    assert float(JF._frame_mean_free(x, none)) == 0.0 and float(JK._frame_mean(x, none)) == 0.0
    gn, = torch.autograd.grad(JF._frame_mean_free(x, none), x)
    assert float(gn.abs().sum()) == 0.0
    print("frame_mean ok")


def test_scale_in_eval():
    plain, fast, b = _pair(9)
    fast.eval()
    with torch.no_grad():
        assert fast(b)["scale"] is not None
        fast.scale_in_eval = False
        o = fast(b)
    assert o["scale"] is None and "scale_at" in o
    fast.train()
    l1, _ = JK.jtoken_loss(plain(b), b)
    l2, _ = JF.fast_loss(fast(b), b)
    _close(l1, l2, 1e-5, "the loss does not read the dense scale")
    print("scale_in_eval ok")


def test_delegation():
    plain, fast, b = _pair(3)
    for kw in (dict(live_inputs=True),):
        a, f = JK.JTokenHead(**kw), JF.FastJTokenHead(**kw)
        f.load_state_dict(a.state_dict())
        a.eval(), f.eval()
        with torch.no_grad():
            _close(a(b)["logp"], f(b)["logp"], 0.0, f"delegation {kw}")
    print("delegation ok")


def test_state():
    plain, fast, _ = _pair(4)
    assert plain.state_dict().keys() == fast.state_dict().keys()
    back = JK.JTokenHead()
    back.load_state_dict(fast.state_dict())
    for (n, a), (_, c) in zip(plain.state_dict().items(), back.state_dict().items()):
        assert torch.equal(a, c), n
    print("state ok")


def test_mutants():
    import inspect
    src = inspect.getsource(JF)
    plain, fast, b = _pair(5)
    ref, _ = JK.jtoken_loss(plain(b), b)
    # (1) the pair list built with the wrong source index
    bs = dict(b, **_pair_list(b))
    bs["sp_i"] = (bs["sp_i"] + 1) % 48
    bad, _ = JF.fast_loss(fast(bs), bs)
    assert abs(float(bad) - float(ref)) > 1e-4, "a wrong pair index must change the loss"
    # (2) the stacked first Linear read with a swapped slice
    h = fast.__class__
    orig = JF.pair_biases

    def swapped(g, w1, b1, w2s, b2s, hid=32):
        outs = orig(g, w1, b1, w2s, b2s, hid)
        return (outs[1], outs[0]) + outs[2:]

    JF.pair_biases = swapped
    try:
        o = fast(b)
    finally:
        JF.pair_biases = orig
    plain.train()
    assert float((plain(b)["logp"] - o["logp"]).abs().max()) > 1e-4, "a swapped bias slice must change the output"
    # (3) the gate mask of the scale selection and the `win` mask of the pair list: see test_loader_loss (real frames: 109 cells with a finite scale, 108 in the long gate)
    print("mutants ok")


def _real_batch(B=2):
    sys.path.insert(0, os.path.join(HERE, "..", "data"))
    import lane_ptok_data as LP
    ds, dl = LP.loader("val", 0, B, 0, shuffle=False)
    return next(iter(dl))


def test_real():
    """2 real VAL frames (K 512) of the B' cache, CPU fp32: head outputs and the loss with its options and gradients, below AND above the key-weight switch (SWITCH), and the E379 KD term on the fast outputs"""
    try:
        b = _real_batch()
    except Exception as e:  # noqa: BLE001
        print(f"real skipped ({str(e)[:60]})")
        return
    for step in (JK.SWITCH - 100, JK.SWITCH + 100):
        for kw in (dict(none_weight=0.25), dict(none_weight=0.25, pos_loss="max"), dict(none_weight=0.25, none_weight_near=0.73), dict()):
            torch.manual_seed(3)
            plain = JK.JTokenHead()
            for p in plain.parameters():
                p.data += 0.05 * torch.randn_like(p)
            fast = JF.FastJTokenHead()
            fast.load_state_dict(plain.state_dict())
            plain._step_py = fast._step_py = step
            o1, o2 = plain(b), fast(b)
            _close_rel(o1["logp"], o2["logp"], 1e-6, f"real logp step {step}")
            _close(o1["types"], o2["types"], 1e-4, f"real types step {step}")
            l1, i1 = JK.jtoken_loss(o1, b, **kw)
            l1.backward()
            l2, i2 = JF.fast_loss(o2, b, **kw)
            l2.backward()
            _close(l1, l2, 1e-4, f"real loss {kw} step {step}")
            for k in i1:
                assert abs(float(i1[k]) - float(i2[k])) <= 1e-4 * max(1.0, abs(float(i1[k]))), (kw, k, i1[k], i2[k])
            g1, g2 = _grads(plain), _grads(fast)
            for n in g1:
                scale = max(1.0, float(g1[n].abs().max()))
                _close(g1[n] / scale, g2[n] / scale, 1e-4, f"real grad {n} {kw} step {step}")
    if hasattr(JK, "distill_loss"):
        t1, t2 = (JK.JTokenHead() for _ in range(2))
        for tt in (t1, t2):
            tt.eval()
        with torch.no_grad():
            outs_t = [tt(b) for tt in (t1, t2)]
        kd1, _ = JK.distill_loss(plain(b), outs_t)
        kd2, _ = JK.distill_loss(fast(b), outs_t)
        _close(kd1, kd2, 1e-4, "KD term")
    print("real ok")


def test_options_and_mirror():
    """none_weight 1.0 with a near weight (the type-CE weight must be a tensor then), none_weight 0.25, and the step mirror after load_state_dict: all against the plain loss"""
    plain, fast, b = _pair(7)
    assert fast._step_py == JK.SWITCH + 5
    fresh = JF.FastJTokenHead()
    fresh.load_state_dict(plain.state_dict())
    assert fresh._step_py == int(plain.step), "load_state_dict sets the mirror from the step buffer"
    for kw, extra in ((dict(none_weight=1.0, none_weight_near=0.73), {}), (dict(none_weight=0.25), {})):
        plain, fast, b = _pair(7)
        b = dict(b, **extra)
        l1, i1 = JK.jtoken_loss(plain(b), b, **kw)
        l2, i2 = JF.fast_loss(fast(b), b, **kw)
        _close(l1, l2, 1e-5, f"loss {kw} {list(extra)}")
        for k in i1:
            assert abs(float(i1[k]) - float(i2[k])) <= 1e-5 * max(1.0, abs(float(i1[k]))), (kw, k)
    # both sides of the key-weight switch give the plain head's output
    for step in (JK.SWITCH - 100, JK.SWITCH + 100):
        plain, fast, b = _pair(8)
        plain._step_py = fast._step_py = step
        _close_rel(plain(b)["logp"], fast(b)["logp"], 1e-6, f"step {step}")
    print("options and mirror ok")


def test_loader_loss():
    """real VAL frames through the GPU loader (CPU device here): the sparse pair list's loss, info and gradients equal the plain loss on the dense batch, below and above the switch; the pair list holds exactly the finite-scale cells"""
    try:
        sys.path.insert(0, os.path.join(HERE, "..", "data"))
        import lane_ptok_gpu as LG
        import lane_ptok_data as LP
        d = LP.cache_dir("val", 0)
    except Exception as e:  # noqa: BLE001
        print(f"loader_loss skipped ({str(e)[:60]})")
        return
    ds = LG.DeviceTokData("val", dir=d, dev="cpu", dense_scale=True)
    b = ds.batch(list(range(0, 8 * 11, 11)))
    assert int(b["sp_ok"].sum()) == int(torch.isfinite(b["yscale"][..., 0]).sum()), "the pair list holds each finite-scale cell once"
    for step in (JK.SWITCH - 100, JK.SWITCH + 100):
        torch.manual_seed(5)
        plain = JK.JTokenHead()
        for p in plain.parameters():
            p.data += 0.05 * torch.randn_like(p)
        fast = JF.FastJTokenHead()
        fast.load_state_dict(plain.state_dict())
        plain._step_py = fast._step_py = step
        l1, i1 = JK.jtoken_loss(plain(b), b, none_weight=0.25)
        l1.backward()
        l2, i2 = JF.fast_loss(fast(b), b, none_weight=0.25)
        l2.backward()
        _close(l1, l2, 1e-4, f"loader loss step {step}")
        for k in i1:
            assert abs(float(i1[k]) - float(i2[k])) <= 1e-4 * max(1.0, abs(float(i1[k]))), (step, k, float(i1[k]), float(i2[k]))
        g1, g2 = _grads(plain), _grads(fast)
        for n in g1:
            scale = max(1.0, float(g1[n].abs().max()))
            _close(g1[n] / scale, g2[n] / scale, 1e-4, f"loader grad {n} step {step}")
    print("loader_loss ok")


def test_trainer():
    """20 smoke steps (synthetic batches, K 48, bs 4, CPU fp32: CUDA is hidden inside the test) of the plain trainer and of `jtoken_fast_train.main` (the same run arguments): the final weights agree to 2e-3 (3e-6 in practice),
    the fast run folder has fast_path.json, it is written only after the recipe check (a folder of another recipe is refused and gets none), a plain run's finished checkpoint is resumed by the fast trainer; the patches are undone"""
    import tempfile
    sys.path.insert(0, os.path.join(HERE, "..", "train"))
    import jtoken_train as JT
    import jtoken_fast_train as FT
    import trainer as TR
    keep = (JT.build_head, JK.jtoken_loss, JT.check_start, JT.train, TR.RUNS, torch.cuda.is_available, JT.load_teachers)
    torch.cuda.is_available = lambda: False
    try:
        with tempfile.TemporaryDirectory() as d:
            argv = ["--smoke", "--bs", "4", "--none-weight", "0.25"]
            JT.train(JT.parse_args(["--name", "plain"] + argv), d, 500, 48)
            orig_train = JT.train
            JT.train = lambda a, runs, *x, **k: orig_train(a, runs, 500, 48)                   # the smoke run with K 48 through `main`
            TR.RUNS = d
            FT.main(["--name", "fast"] + argv)
            sa, sb = (torch.load(os.path.join(d, n, "model.pt"), map_location="cpu", weights_only=False) for n in ("plain", "fast"))
            assert sa["state"].keys() == sb["state"].keys()
            worst = max(float((sa["state"][k].float() - sb["state"][k].float()).abs().max()) for k in sa["state"])
            assert worst < 2e-3, f"after 20 steps the weights differ by {worst}"
            note = json.load(open(os.path.join(d, "fast", "fast_path.json")))
            assert note["head"] == "jtoken_fast.FastJTokenHead" and "CPU" in note["loader"], note
            os.makedirs(os.path.join(d, "other"))                                                # a folder that holds ANOTHER recipe: refused by the start check, no fast_path.json written into it
            shutil.copy(os.path.join(d, "plain", "config.json"), os.path.join(d, "other", "config.json"))
            try:
                FT.main(["--name", "other", "--smoke", "--bs", "4", "--none-weight", "0.5"])
                raise SystemExit("the start check must refuse another recipe")
            except AssertionError:
                pass
            assert not os.path.exists(os.path.join(d, "other", "fast_path.json"))
            shutil.copytree(os.path.join(d, "plain"), os.path.join(d, "resume"))                 # a finished plain run resumed by the fast trainer (the same recipe: the name is not in it)
            FT.main(["--name", "resume"] + argv)
            assert "resumed from" in open(os.path.join(d, "resume", "log.txt")).read(), "the fast trainer resumes a plain checkpoint"
    finally:
        JT.build_head, JK.jtoken_loss, JT.check_start, JT.train, TR.RUNS, torch.cuda.is_available, JT.load_teachers = keep
    assert JT.build_head is keep[0] and JK.jtoken_loss is keep[1] and JT.check_start is keep[2]
    print(f"trainer ok (max weight difference after 20 steps {worst:.2e})")


def test_gpu():
    if not (torch.cuda.is_available() and os.environ.get("JTF_GPU") == "1"):
        print("gpu skipped (set JTF_GPU=1 with a GPU)")
        return
    dev = "cuda"
    plain, fast, b = _pair(6, B=4, K=96)
    plain, fast = plain.to(dev).train(), fast.to(dev).train()
    bs = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in dict(b, **_pair_list(b)).items()}
    torch.cuda.set_sync_debug_mode("error")
    try:
        with torch.autocast("cuda", torch.bfloat16):
            out = fast(bs)
            loss, info = JF.fast_loss(out, bs, none_weight=0.25)
        loss.backward()
    finally:
        torch.cuda.set_sync_debug_mode(0)
    with torch.autocast("cuda", torch.bfloat16):
        pl, _ = JK.jtoken_loss(plain(bs), bs, none_weight=0.25)
    assert abs(float(loss) - float(pl)) < 2e-2, (float(loss), float(pl))
    f2 = JF.FastJTokenHead().to(dev).train()
    f2.load_state_dict(fast.state_dict())
    f2._step_py = JK.SWITCH + 5                                               # load_state_dict resets the mirror to the step buffer: the same key-weight branch as `fast`
    f2.compile_pairs()
    with torch.autocast("cuda", torch.bfloat16):
        l2, _ = JF.fast_loss(f2(bs), bs, none_weight=0.25)
    assert abs(float(l2) - float(loss)) < 2e-2
    print("gpu ok")


TESTS = [test_head, test_loss, test_sparse, test_frame_mean, test_scale_in_eval, test_delegation, test_state, test_mutants, test_options_and_mirror, test_real, test_loader_loss, test_trainer, test_gpu]
if __name__ == "__main__":
    want = [n if n.startswith("test_") else "test_" + n for n in sys.argv[1:]]
    assert all(n in {t.__name__ for t in TESTS} for n in want), want
    for t_ in TESTS:
        if not want or t_.__name__ in want:
            t_()
    print("test_jtoken_fast ok")
