import io

p = "evaluation/jstack.py"
s = io.open(p, encoding="utf-8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:90], s.count(a))
    s = s.replace(a, b)


# 1. a pure comparison of the node outputs: missing keys are errors, NaN fails, nothing compared fails
rep('''def sha256(p):''', '''def node_diff(nd, nb, layered):
    """-> ({key: largest absolute difference}, ok) of the pair model's node outputs `nd` against `QGraph.node_stage` outputs `nb` of B': `e` and `pos` must be in both (a missing key is an error, not a pass), `e_raw` must be in both when B' has the layer
    (`layered`), the differences must be finite and at most 1e-5"""
    worst, ok = {}, True
    for k in ("e", "pos") + (("e_raw",) if layered else ()):
        assert k in nd and k in nb, f"node output {k!r} missing (pair model: {sorted(nd)}, B': {sorted(nb)}): nothing compared"
        v = float((nd[k].float() - nb[k].float()).abs().max())
        worst[k] = max(worst.get(k, 0.0), v) if v == v else float("nan")
        ok = ok and v == v and v <= 1e-5
    return worst, ok


def sha256(p):''')
rep('''    worst = dict(e=0.0, pos=0.0, e_raw=0.0)
    n = 0
    with torch.no_grad():
        for b in dl:
            x = b["x"].to(dev).float()
            P, _ = m.jmodel(x, joint=False)
            nd = m.nodes(P["feat2"], P["feat1"], P["feat05"], P["road2"], P["centre1"])
            nb = B.node_stage(B.pyr(x))
            for k in worst:
                if k in nb:
                    worst[k] = max(worst[k], float((nd[k].float() - nb[k].float()).abs().max()))
            n += x.shape[0]
            if n >= a.n:
                break
    ok = all(v <= 1e-5 for v in worst.values())
    lp = layer_params(a.g, a.seed)
    print(f"NODE CHECK {run}: {n} frames, layer {'mnms %s' % (lp,) if lp else 'none'}, max |difference| e {worst['e']:.2e}, pos {worst['pos']:.2e}, e_raw {worst['e_raw']:.2e}: {'PASS' if ok else 'FAIL'}")''',
    '''    lp = layer_params(a.g, a.seed)
    worst, ok_all = dict(e=0.0, pos=0.0, e_raw=0.0), True
    n = 0
    with torch.no_grad():
        for b in dl:
            x = b["x"].to(dev).float()
            P, _ = m.jmodel(x, joint=False)
            nd = m.nodes(P["feat2"], P["feat1"], P["feat05"], P["road2"], P["centre1"])
            nb = B.node_stage(B.pyr(x))
            w, ok_b = node_diff(nd, nb, lp is not None)
            ok_all = ok_all and ok_b
            for k, v in w.items():
                worst[k] = max(worst[k], v) if v == v else float("nan")
            n += x.shape[0]
            if n >= a.n:
                break
    ok = ok_all and n > 0
    print(f"NODE CHECK {run}: {n} frames, layer {'mnms %s' % (lp,) if lp else 'none'}, max |difference| e {worst['e']:.2e}, pos {worst['pos']:.2e}, e_raw {worst['e_raw']:.2e}: {'PASS' if ok else 'FAIL'}")''')
# 2. dump: the original loader is captured once and restored
rep('''    orig = JE.load_model
    JE.load_model = lambda run, dev: model_for(run, dev, orig)                                              # `jends_eval.dump` calls `load_model(a.run, dev)` of its own module: an in-process patch, no frozen file is edited
    JE.dump(NS(run=a.run, thr=0.5, max=0))''', '''    orig = JE.load_model
    assert not getattr(orig, "_jstack_patched", False), "jends_eval.load_model is already patched by jstack: wrapping it twice would damp the layer twice"
    patched = lambda run, dev: model_for(run, dev, orig)  # noqa: E731
    patched._jstack_patched = True
    JE.load_model = patched                                                                                  # `jends_eval.dump` calls `load_model(a.run, dev)` of its own module: an in-process patch, no frozen file is edited
    try:
        JE.dump(NS(run=a.run, thr=0.5, max=0))
    finally:
        JE.load_model = orig''')
# 3. check-arrays: the key sets must be equal
rep('''            for k, v in mine.items():
                if not torch.equal(v, nsrc[k] if k.startswith("nodes.") else src[k]):''', '''            if set(mine) != set(src):
                r["ok"] = False
                r["notes"].append(f"tensor key sets differ from E324 s: {sorted(set(mine) ^ set(src))[:4]}")
            for k, v in mine.items():
                if k not in src or not torch.equal(v, nsrc[k] if k.startswith("nodes.") else src[k]):''')
# selftest additions
rep('''    print("jstack selftest ok")''', '''    # 7. the node comparison: equal passes, a difference above 1e-5 fails, a NaN fails, a missing key is an error (not a pass), e_raw only needed with the layer
    t = lambda v: torch.full((1, 2, 2), float(v))  # noqa: E731
    good = dict(e=t(1.0), pos=t(2.0))
    assert node_diff(good, dict(good), False) == (dict(e=0.0, pos=0.0), True)
    assert node_diff(dict(good, e_raw=t(3.0)), dict(good, e_raw=t(3.0)), True)[1]
    assert not node_diff(dict(good, e=t(1.001)), good, False)[1] and node_diff(dict(good, e=t(1.000001)), good, False)[1] is True
    assert not node_diff(dict(good, pos=t(float("nan"))), good, False)[1] and not node_diff(good, dict(good, e=t(float("nan"))), False)[1]
    for nd_, nb_, lay in ((dict(e=t(1.0)), good, False), (good, dict(e=t(1.0)), False), (good, good, True), (dict(good, e_raw=t(1.0)), good, True)):
        try:
            node_diff(nd_, nb_, lay)
        except AssertionError:
            continue
        raise AssertionError("node_diff accepted a missing key")
    print("jstack selftest ok")''')
io.open(p, "w", encoding="utf-8").write(s)
print("ok")
