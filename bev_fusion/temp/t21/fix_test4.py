p = "tests/test_stream_temporal.py"
s = open(p, encoding="utf-8").read()

def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (s.count(a), a)
    s = s.replace(a, b)

rep('''    return {f"A pair stream = batch of two frames ({tag})": gap(run(p, mk([1]), r0["state"], SEGS[0]), run(p, mk([0, 1]), r0["state"], SEGS[0])) < TOL}
''', '''    w = build(**kw)                                                   # the window model, the same weights: two frames are its window with K = 1
    return {f"A pair stream = batch of two frames ({tag})": gap(run(p, mk([1]), r0["state"], SEGS[0]), run(p, mk([0, 1]), r0["state"], SEGS[0])) < TOL,
            f"A2 pair batch of two frames = the window model on the same two frames ({tag})": gap(run(p, mk([0, 1])), run(w, mk([0, 1])), buf=False) == 0}
''')
rep('''    ok = {f"B1 recurrent stream of three frames = batch unroll ({tag})": gap(stream(r, [0, 1, 2])[2], run(r, mk([0, 1, 2]))) < TOL}
''', '''    ok = {f"B1 recurrent stream of three frames = batch unroll ({tag})": gap(stream(r, [0, 1, 2])[2], run(r, mk([0, 1, 2]))) < TOL}
    if kw:                                                            # the motion block reads the two raw BEVs, the frame before the current one
        ref = run(build(**kw), mk([1, 2]))["out"][0.0]
        got = run(r, mk([0, 1, 2]))["out"][0.0]
        ok[f"B6 the motion field of the recurrent batch = the window model's on the last two frames ({tag})"] = \
            all(torch.equal(got[k], ref[k]) if torch.is_tensor(got[k]) else all(torch.equal(got[k][j], ref[k][j]) for j in ref[k]) for k in ("v_hat", "v_levels"))
''')
open(p, "w", encoding="utf-8", newline="").write(s)
