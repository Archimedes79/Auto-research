p = "tests/test_stream_temporal.py"
s = open(p, encoding="utf-8").read()
a = s.index("def core_checks():")
b = s.index("# ------------------------------------------------------------------------------------------------ C gradient")
new = '''def pair_checks(tag, kw):
    p = build(cam_buffer="pair", **kw)
    r0 = run(p, mk([0]))                                              # the previous call: frame 0, cold
    return {f"A pair stream = batch of two frames ({tag})": gap(run(p, mk([1]), r0["state"], SEGS[0]), run(p, mk([0, 1]), r0["state"], SEGS[0])) < TOL}


def recurrent_checks(tag, kw):
    r = build(cam_buffer="recurrent", **kw)
    ok = {f"B1 recurrent stream of three frames = batch unroll ({tag})": gap(stream(r, [0, 1, 2])[2], run(r, mk([0, 1, 2]))) < TOL}
    s = stream(r, [0, 1, 2]) + [None] * 3                             # frames 3 - 5 follow; the step to frame 3 is the keyframe step less its own chain
    dt_o = torch.full((B,), 0.5) - sum((x[0] for x in SEGS[3] + SEGS[4]), torch.zeros(B))
    s[3] = run(r, mk([3]), s[2]["state"], [(dt_o, VG, WG)])
    s[4] = run(r, mk([4]), s[3]["state"], SEGS[3])
    s[5] = run(r, mk([5]), s[4]["state"], SEGS[4])
    k1 = run(r, mk([0, 1, 2]))
    k2 = run(r, mk([3, 4, 5]), k1["state"], (torch.full((B,), 0.5), VG, WG))
    ok[f"B2 keyframe batch with the previous state and the gap step = the stream of six frames ({tag})"] = gap(s[5], k2) < TOL
    cold = run(r, mk([3, 4, 5]), {k: v for k, v in k1["state"].items() if k != "buf"}, (torch.full((B,), 0.5), VG, WG))
    ok[f"B3 the carried buffer matters ({tag})"] = gap(k2, cold) > 1e-3
    return ok


def buffer_rules():
    bogus = {"raw": torch.randn(B, 16, 32, 32), "fused": torch.randn(B, 16, 32, 32)}
    r = build(cam_buffer="recurrent")
    p = build(cam_buffer="pair")
    r0 = run(p, mk([0]))
    return {"B4 recurrent batch without a step ignores a buffer (cold)": gap(run(r, mk([0, 1, 2]), {"buf": bogus}), run(r, mk([0, 1, 2]))) < 1e-7,
            "B5 pair batch never reads a buffer": gap(run(p, mk([0, 1]), dict(r0["state"], buf=bogus), SEGS[0]), run(p, mk([0, 1]), r0["state"], SEGS[0])) < 1e-7}


def core_checks():
    """the equalities and rules the mutants must break: name -> bool (an exception counts as a failed group)"""
    ok = {}
    for fn, *args in [(pair_checks, t, k) for t, k in KEYS] + [(recurrent_checks, t, k) for t, k in KEYS] + [(buffer_rules,), (grad_checks,)]:
        try:
            ok.update(fn(*args))
        except Exception as e:                                        # noqa: BLE001 - a planted bug that crashes is caught as well
            ok[f"{fn.__name__}{args[:1]} raised {type(e).__name__}"] = False
    return ok


'''
s = s[:a] + new + s[b:]
open(p, "w", encoding="utf-8", newline="").write(s)
