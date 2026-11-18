"""E387 (S2b): `jpast_eval.py time` (promoted from the scratch temp/e368/time_s2.py, a second use) + its tests, in the scratch tree. usage: python patch_s2b_eval.py <tree root>"""
import os
import sys

root = os.path.abspath(sys.argv[1])


def edit(rel, pairs):
    p = os.path.join(root, rel)
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (rel, s.count(old), old[:90])
        s = s.replace(old, new)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(s)


TIME = '''def time_head(m, batches, dev, repeats):
    """-> (warm_ms, cold_ms): the per-frame time of the HEAD ALONE (batch 1, bf16 autocast on the GPU, CUDA events with a sync in front and behind: `jtiming.timed`) over `repeats` passes of `batches` (after one untimed pass) in stream order: the past of frame t = the head's own states of
    frame t - 1 moved by the stored pose, exactly what the stream read-out does, without its kill-measure hooks. A frame with a past (or any frame of a head without a block) is warm; the first frame of a scene (a block, no past) is cold"""
    block = getattr(m, "past_block", None)
    cuda = dev == "cuda"
    warm, cold = [], []
    for rep in range(repeats + 1):
        out_prev = fr_prev = None
        row_warm, row_cold = [], []
        for b in batches:
            hp = bool(b["has_prev"][0, 0])
            frd = JKT.to_device(b["frames"][0], dev)
            R_, t_ = b["rel_R"][:, 0].to(dev), b["rel_t"][:, 0].to(dev)
            past = {}
            if block is not None and hp and out_prev is not None:
                past = JP.make_past(out_prev, fr_prev, frd, R_, t_, m._step_py >= JK.SWITCH)
            with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=cuda):
                out, ms = JTG.timed(lambda: (m(frd, past) if (block is not None and past) else m(frd)), cuda)
            (row_warm if (past or block is None) else row_cold).append(ms)
            out_prev, fr_prev = out, frd
        if rep:
            warm += row_warm
            cold += row_cold
    return warm, cold


def time_cmd(a, runs=JE.RUNS):
    """E368 / E387 inference cost: the head alone of each run in `--runs` (a run WITH a block first, the run without one second: the block cost = the difference of the medians) on the first `--frames` VAL frames of the stream, `--repeats` passes; host load and GPU state before and after (the after-sample waits 2.5 s: `token_time_ens.after_state`) -> a json
    `runs/<first run>/past_timing_<UTC>.json` with the QUIET verdict (`usable_for_gcost`)"""
    import token_time_ens as TT
    import lane_past_data as LPD
    ok, why = (False, "--cpu") if a.cpu else JE.gpu_free()
    if not a.cpu and not ok:
        sys.exit(f"STOP: no GPU timing ({why})")
    dev = "cpu" if a.cpu else "cuda"
    res = dict(device=dev, frames=a.frames, repeats=a.repeats, gpu_rule=why, arms={})
    load0, gpu0 = JTG.load1(), (None if a.cpu else JTG.gpu_state())
    for run in a.runs.split(","):
        m, args = JPT.load_model(run, dev, runs)
        m.eval()
        kw = JE.cache_kw(args)
        _, dl = LPD.loader("val", kw["seed"], 1, 0, ident=kw["ident"], K=kw["K"], shuffle=False, stream=True, **({"cache": kw["cache"]} if kw.get("cache") else {}))
        batches = []
        for b in dl:
            batches.append(b)
            if len(batches) == a.frames:
                break
        warm, cold = time_head(m, batches, dev, a.repeats)
        block = getattr(m, "past_block", None)
        res["arms"][run] = dict(params=JK.count_params(m), params_block=JK.count_params(block) if block is not None else 0, has_block=block is not None, n_timed=len(warm), median_ms=float(np.median(warm)), p90_ms=float(np.percentile(warm, 90)),
                                cold_frames=len(cold), cold_median_ms=float(np.median(cold)) if cold else None)
    load1, gpu1 = TT.after_state(JTG, a.cpu)
    quiet, line = JTG.quiet_verdict(load0, load1, dict(util_max=0.0) if a.cpu else gpu0, dict(util_max=0.0) if a.cpu else gpu1)
    res.update(host=line, quiet=bool(quiet), usable_for_gcost=bool(quiet and dev == "cuda"))
    arms = list(res["arms"].values())
    if len(arms) == 2 and arms[0]["has_block"] and not arms[1]["has_block"]:
        res["block_cost_ms"] = arms[0]["median_ms"] - arms[1]["median_ms"]
    print(json.dumps(res, indent=1))
    first = a.runs.split(",")[0]
    print("written", os.path.relpath(JE._write_stamped(os.path.join(runs, first), "past_timing" + ("_cpu" if a.cpu else ""), res), JE.ROOT))
    return res


def main(argv=None):'''

edit("evaluation/jpast_eval.py", [
    ("import jpast_train as JPT  # noqa: E402\n", "import jpast_train as JPT  # noqa: E402\nimport jtiming as JTG  # noqa: E402\n"),
    ("def main(argv=None):", TIME),
    ('    a = ap.parse_args(argv)\n    {"dump": dump, "use": use}[a.cmd](a)', '    s = sub.add_parser("time")\n    s.add_argument("--runs", required=True, help="a run WITH a block first, optionally a run without one second (the block cost)")\n    s.add_argument("--frames", type=int, default=24)\n    s.add_argument("--repeats", type=int, default=4)\n    s.add_argument("--cpu", action="store_true")\n    a = ap.parse_args(argv)\n    {"dump": dump, "use": use, "time": time_cmd}[a.cmd](a)'),
])

edit("training/grid/test_jpast_train.py", [
    ('    ("jpast_eval.py", "the first 5 frames are 4", "FIRST = 5 ", "FIRST = 4 "),',
     '    ("jpast_eval.py", "the first 5 frames are 4", "FIRST = 5 ", "FIRST = 4 "),\n'
     '    ("jpast_eval.py", "time: a frame with a past is timed as cold", "(row_warm if (past or block is None) else row_cold)", "(row_cold if (past or block is None) else row_warm)"),\n'
     '    ("jpast_eval.py", "time: the untimed pass is timed", "        if rep:\\n            warm += row_warm", "        if True:\\n            warm += row_warm"),\n'
     '    ("jpast_eval.py", "time: the state of the previous frame is not carried", "                past = JP.make_past(out_prev, fr_prev, frd, R_, t_, m._step_py >= JK.SWITCH)", "                past = {}"),'),
    ('    frames = [dict(JK.with_tokens_np(JK.synthetic_batch(1, K, seed=10 + i)), token=[f"t{i}"], row=torch.tensor([i])) for i in range(n)]\n    batches = lambda: [dict(frames=[frames[i]], rel_R=R[i][None, None], rel_t=t[i][None, None], has_prev=torch.tensor([[hp[i]]]), idx=torch.tensor([[i]])) for i in range(n)]  # noqa: E731\n',
     '    frames = [dict(JK.with_tokens_np(JK.synthetic_batch(1, K, seed=10 + i)), token=[f"t{i}"], row=torch.tensor([i])) for i in range(n)]\n    batches = lambda: [dict(frames=[frames[i]], rel_R=R[i][None, None], rel_t=t[i][None, None], has_prev=torch.tensor([[hp[i]]]), idx=torch.tensor([[i]])) for i in range(n)]  # noqa: E731\n'
     '    warm, cold = E.time_head(m, batches(), "cpu", 2)                                                     # the timing loop: warm = frames with a past, cold = the scene starts, the first pass is not counted\n'
     '    assert len(warm) == 2 * sum(hp) and len(cold) == 2 * (n - sum(hp)) and min(warm + cold) > 0, f"{label}: time_head counts {len(warm)} {len(cold)}"\n'
     '    mz0 = T.build_head(dict(no_past=True)).eval()\n'
     '    warm0, cold0 = E.time_head(mz0, batches(), "cpu", 1)\n'
     '    assert len(warm0) == n and cold0 == [], f"{label}: a head without a block is all warm"\n'),
])
print("patched", root)
