p = "tests/test_stream_temporal.py"
s = open(p, encoding="utf-8").read()

def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (s.count(a), a)
    s = s.replace(a, b)

rep('''def core_checks():''', '''def advance_checks():
    ok = {}
    for kind in ("pair", "recurrent"):
        m = build(cam_buffer=kind, **MOTION)
        r0 = run(m, mk([0]))
        with torch.no_grad():
            cold, warm = m.advance(mk([0])), m.advance(mk([1]), r0["state"], SEGS[0])
        full = run(m, mk([1]), r0["state"], SEGS[0])["state"]
        same = lambda a, b: all(torch.equal(a[k][kk], b[k][kk]) for k in ("cam", "buf") for kk in b[k])      # noqa: E731
        ok[f"D advance (lift + temporal block only) gives the state of the full step, cold and buffered ({kind})"] = same(cold, r0["state"]) and same(warm, full)
    return ok


def core_checks():''')
rep('''+ [(buffer_rules,), (grad_checks,)]:''', '''+ [(buffer_rules,), (advance_checks,), (grad_checks,)]:''')
rep('''check(refuses(lambda: run(build(cam_buffer="pair"), mk([0, 1, 2]))), "a 'pair' batch of three frames is refused")
''', '''check(refuses(lambda: run(build(cam_buffer="pair"), mk([0, 1, 2]))), "a 'pair' batch of three frames is refused")
check(refuses(lambda: build().advance(mk([0]))), "advance on a window model is refused")
''')
rep('''  D  cold start in the stream: finite, equal to the window model with one frame (no motion block); with the motion block the current frame stands
     in as its older frame with a zero step; a buffered stream step without `step` and a 'pair' batch of three frames are refused.''', '''  D  cold start in the stream: finite, equal to the window model with one frame (no motion block); with the motion block the current frame stands
     in as its older frame with a zero step; a buffered stream step without `step` and a 'pair' batch of three frames are refused;
     `advance` (a stream frame that needs no output: lift + temporal block only) returns the state of the full step.''')
open(p, "w", encoding="utf-8", newline="").write(s)
