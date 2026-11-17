p = "evaluation/jpast_eval.py"
s = open(p, encoding="utf-8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:80])
    s = s.replace(a, b)


rep('''    donor = states = frame_of = None
    if a.mode == "shuffled":
        states = {i: (out["x"].cpu(), out["w"].cpu()) for i, _, out, _ in iter_outputs(m, dl, "stream", dev=dev)}
        donor, frame_of = donor_plan(ds.prev), (lambda j: LP.collate([ds.tok[j]]))''', '''    donor = states = frame_of = None
    if a.split == "val" and a.mode != "off" and not args.get("no_past"):                  # the states of a whole stream pass: the past of mode shuffled, the other scene's past that u compares with in the other modes (TRAIN: u_none only)
        states = {i: (out["x"].cpu(), out["w"].cpu()) for i, _, out, _ in iter_outputs(m, dl, "stream", dev=dev)}
        donor, frame_of = donor_plan(ds.prev), (lambda j: LP.collate([ds.tok[j]]))''')
rep('''    used = sum(1 for n in use["n"] if n)
    u_txt = f"u {math.sqrt(sum(use['d2']) / max(sum(use['x2']), 1e-30)):.4f} over {used} frames with a past" if used else "no past read (mode off or arm z)"''', '''    used, used_o = sum(1 for n in use["n"] if n), sum(1 for n in use["n_other"] if n)
    u_txt = (f"u_none {math.sqrt(sum(use['d2_none']) / max(sum(use['x2']), 1e-30)):.4f} over {used} frames with a past" + (f", u {math.sqrt(sum(use['d2_other']) / max(sum(use['x2_other']), 1e-30)):.4f} (against another scene's past) over {used_o}" if used_o else "")
             if used else "no past read (mode off or arm z)")''')
rep('''    """u of the kill measure overall and by stream position, and (with the dump of mode off) what the past changed: the dumps are the dicts of `dump` (meta + frames) -> dict"""
    u = stream["meta"]["use"]
    d2, x2, pos, hp, sq, sn = (np.asarray(u[k], float) for k in ("d2", "x2", "pos", "has_prev", "state_ss", "state_n"))
    ratio = lambda m: float(math.sqrt(d2[m].sum() / max(x2[m].sum(), 1e-30))) if m.any() and x2[m].sum() > 0 else float("nan")  # noqa: E731
    rep = dict(run=stream["meta"]["run"], split=stream["meta"]["split"], mode=stream["meta"].get("mode", "stream"), frames=len(d2), frames_with_past=int(hp.sum()), u=ratio(hp > 0),
               u_by_position={f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+": ratio((pos >= lo) & (pos <= hi) & (hp > 0)) for lo, hi in BUCKETS},''', '''    """the kill measures overall and by stream position, and (with the dump of mode off) what the past changed: the dumps are the dicts of `dump` (meta + frames) -> dict. u = RMS(update with the past - update with the past of ANOTHER scene) / RMS(x) (the frames
    that have an other-scene past: VAL, a block, a mode with a past; NaN else), u_none = the same against the none key only (any content-free attention satisfies it: no test of use)"""
    u = stream["meta"]["use"]
    d2n, x2, d2o, x2o, pos, hp, sq, sn = (np.asarray(u[k], float) for k in ("d2_none", "x2", "d2_other", "x2_other", "pos", "has_prev", "state_ss", "state_n"))
    ratio = lambda d2, xx, m: float(math.sqrt(d2[m].sum() / max(xx[m].sum(), 1e-30))) if m.any() and xx[m].sum() > 0 else float("nan")  # noqa: E731
    bucket = {f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+": (pos >= lo) & (pos <= hi) & (hp > 0) for lo, hi in BUCKETS}
    has_o = np.asarray(u["n_other"], float) > 0
    rep = dict(run=stream["meta"]["run"], split=stream["meta"]["split"], mode=stream["meta"].get("mode", "stream"), frames=len(d2n), frames_with_past=int(hp.sum()), frames_with_other_past=int(has_o.sum()), u=ratio(d2o, x2o, has_o), u_none=ratio(d2n, x2, hp > 0),
               u_by_position={k: ratio(d2o, x2o, mk & has_o) for k, mk in bucket.items()}, u_none_by_position={k: ratio(d2n, x2, mk) for k, mk in bucket.items()},''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
