p = "training/grid/test_jtoken_fast.py"
t = open(p, encoding="utf-8").read()
# B1: the step mirror after load_state_dict in the gpu test
a = '''    f2.load_state_dict(fast.state_dict())
    f2.compile_pairs()'''
assert t.count(a) == 1
t = t.replace(a, '''    f2.load_state_dict(fast.state_dict())
    f2._step_py = JK.SWITCH + 5                                               # load_state_dict resets the mirror to the step buffer: the same key-weight branch as `fast`
    f2.compile_pairs()''')
# the grep mutant is not a behaviour test: the real-loader test below carries it
a = '''    # (3) the gate mask of the scale selection left out
    assert "cand[:, 1][bi, ii, jj]" in src and "G[:, 1] & torch.isfinite" in src
'''
assert t.count(a) == 1
t = t.replace(a, '''    # (3) the gate mask of the scale selection and the `win` mask of the pair list: see test_loader_loss (real frames: 109 cells with a finite scale, 108 in the long gate)
''')
new = '''def test_options_and_mirror():
    """none_weight 1.0 with a near weight (the type-CE weight must be a tensor then), a `yskip` batch (the E370 rule teacher's key), and the step mirror after load_state_dict: all against the plain loss"""
    plain, fast, b = _pair(7)
    assert fast._step_py == JK.SWITCH + 5
    fresh = JF.FastJTokenHead()
    fresh.load_state_dict(plain.state_dict())
    assert fresh._step_py == int(plain.step), "load_state_dict sets the mirror from the step buffer"
    for kw, extra in ((dict(none_weight=1.0, none_weight_near=0.73), {}), (dict(none_weight=0.25), dict(yskip=torch.rand(3, 2, 48) > 0.5))):
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


def test_trainer():'''
assert t.count("def test_trainer():") == 1
t = t.replace("def test_trainer():", new, 1)
t = t.replace("test_mutants, test_real, test_trainer, test_gpu]", "test_mutants, test_options_and_mirror, test_real, test_loader_loss, test_trainer, test_gpu]")
t = t.replace('mutants (a wrong pair index, no gate mask in the scale selection, the stacked first Linear with a swapped slice: each makes a test fail)', 'mutants (a wrong pair index, the stacked first Linear with a swapped slice: each makes a test fail), options_and_mirror, loader_loss (real frames: gate mask and win mask of the pair list)')
open(p, "w", encoding="utf-8").write(t)

p = "training/grid/test_lane_ptok_gpu.py"
t = open(p, encoding="utf-8").read()
t = t.replace('mutants (the sampler seed drawn without the base-seed draw, a first-wins pair, a dropped win mask: each fails).', 'mutants (the sampler seed drawn without the base-seed draw, a first-wins pair: each fails), gpu (JTF_GPU=1: the same batch on cuda).')
a = '''TESTS = [test_batches, test_real, test_order]'''
assert t.count(a) == 1
t = t.replace(a, '''def test_gpu():
    if not (torch.cuda.is_available() and os.environ.get("JTF_GPU") == "1"):
        print("gpu skipped (set JTF_GPU=1 with a GPU)")
        return
    with tempfile.TemporaryDirectory() as d:
        _toy(d)
        c, g = LG.DeviceTokData("val", dir=d, dev="cpu", dense_scale=True), LG.DeviceTokData("val", dir=d, dev="cuda", dense_scale=True)
        idx = [0, 2, 3, 5]
        a, b = c.batch(idx), g.batch(idx)
        for k in LG.FIELDS + ("ytype", "y", "ylong", "yscale", "sp_b", "sp_i", "sp_j", "sp_y", "sp_ok"):
            _same(a[k], b[k].cpu(), "cuda " + k)
        assert a["token"] == b["token"]
    print("gpu ok")


TESTS = [test_batches, test_real, test_order, test_gpu]''')
open(p, "w", encoding="utf-8").write(t)
print("ok")
