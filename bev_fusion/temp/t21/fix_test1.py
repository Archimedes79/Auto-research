p = "tests/test_stream_temporal.py"
s = open(p, encoding="utf-8").read()

def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (s.count(a), a)
    s = s.replace(a, b)

rep('''def flat(res):
    out = {"features": res["features"]}''', '''def flat(res, buf=True):
    out = {"features": res["features"]}''')
rep('''    for k, v in res["state"].items():
        for kk, t in (v or {}).items():
            out[f"state.{k}.{kk}"] = t
    return out


def gap(a, b):
    """the largest absolute difference of every tensor of two results (outputs, features, states)"""
    fa, fb = flat(a), flat(b)''', '''    for k, v in res["state"].items():
        for kk, t in (v or {}).items():
            if buf or k != "buf":
                out[f"state.{k}.{kk}"] = t
    return out


def gap(a, b, buf=True):
    """the largest absolute difference of every tensor of two results (outputs, features, states; buf=False: not the buffer)"""
    fa, fb = flat(a, buf), flat(b, buf)''')
rep('''def stream(m, idx, first_step=None, steps_after=None):
    """one call per frame of `idx`, the state carried; call i > 0 gets the chain from frame i - 1 to i (the first: `first_step` or none)"""
    res, st = [], None
    for n, i in enumerate(idx):
        step = first_step if n == 0 else SEGS[i - 1]
        res.append(run(m, mk([i]), st, step))''', '''def stream(m, idx):
    """one call per frame of `idx` (consecutive), the state carried; every call after the first gets the chain from the frame before"""
    res, st = [], None
    for n, i in enumerate(idx):
        res.append(run(m, mk([i]), st, SEGS[i - 1] if n else None))''')
rep('''        s = stream(r, [0, 1, 2]) + [None] * 3                         # frames 3 - 5: the first step is the keyframe step less the chain of frame 3
        s[3] = run(''', '''        s = stream(r, [0, 1, 2]) + [None] * 3                         # frames 3 - 5: the first step is the keyframe step less the chain of frame 3
        s[3] = run(''')
# D section
a = s.index('print("D: cold start")')
b = s.index('# ------------------------------------------------------------------------------------------------ E')
s = s[:a] + '''print("D: cold start")
for kind in ("pair", "recurrent"):
    w, m = build(), build(cam_buffer=kind)
    m.load_state_dict(w.state_dict())
    check(gap(run(m, mk([2])), run(w, mk([2])), buf=False) == 0, f"{kind}: cold stream step = the window model with one frame (no past, no motion block), bit for bit")
    mm = build(cam_buffer=kind, **MOTION)
    c = flat(run(mm, mk([2])))
    check(all(torch.isfinite(t).all() for t in c.values()) and float(c["0.0.v_hat."].abs().max()) < 1e3,
          f"{kind} + motion block: cold start (the current frame as its own older frame, zero step) is finite")
    s1 = run(mm, mk([3]), run(mm, mk([2]))["state"], SEGS[2])
    check(all(torch.isfinite(t).all() for t in flat(s1).values()) and "buf" in s1["state"], f"{kind} + motion block: the first buffered step is finite")


def refuses(call):
    try:
        call()
        return False
    except ValueError:
        return True


mm = build(cam_buffer="recurrent", **MOTION)
check(refuses(lambda: run(mm, mk([3]), run(mm, mk([2]))["state"])), "a buffered stream step without `step` is refused")
check(refuses(lambda: run(build(cam_buffer="pair"), mk([0, 1, 2]))), "a 'pair' batch of three frames is refused")

''' + s[b:]
# mutants
rep('''    def cut(*a, **k):
        return TemporalFusion.forward
    return {''', '''    return {''')
rep('''    saved = [(o, a, getattr(o, a)) for o, a, _ in patches]''', '''    saved = [(o, a, vars(o)[a]) for o, a, _ in patches]              # vars(): a staticmethod stays one''')
rep('''check(all(getattr(M.BEVFusionModel, n) is _orig[n] or isinstance(getattr(M.BEVFusionModel, n), type(_orig[n])) for n in _orig), "the mutants are removed again")
check(all(core_checks().values()), "after the mutants the baseline passes again")''', '''check(all(core_checks().values()), "after the mutants the baseline passes again")''')
rep('''_orig = {n: getattr(M.BEVFusionModel, n) for n in ("_buffer_in", "_previous", "_next_buffer", "_step_to_oldest")}


def mutants():
    prev = _orig["_previous"]
''', '''

def mutants():
    prev = M.BEVFusionModel._previous
''')
open(p, "w", encoding="utf-8", newline="").write(s)
