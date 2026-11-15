import io
p = "evaluation/jtiming.py"
s = io.open(p, encoding="utf-8", newline="").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
# docstring usage
rep("    wsl bash tools/env/_py14.sh jtiming.py selftest                                                                                                      # CPU: the statistics, the schedule, the section markers on a toy JEnds\n",
    "    wsl bash tools/env/_py14.sh jtiming.py kernels --runs e312_p_s0,e324_s_s0 [--tag NAME]                                                              # GPU: CUDA launches, memcopies and syncs per stage and per section of one forward (E351 step 1)\n"
    "    wsl bash tools/env/_py14.sh jtiming.py selftest                                                                                                      # CPU: the statistics, the schedule, the section markers on a toy JEnds\n")
# the kernels command
rep("def run(a):\n", '''LAUNCH_NAMES = ("cudaLaunchKernel", "cudaLaunchKernelExC", "cudaMemcpyAsync", "cudaMemcpy", "cudaMemsetAsync", "cudaStreamSynchronize", "cudaDeviceSynchronize")


class ProfMarks(Marks):
    """markers that are zero-length `record_function` ranges, so that they stand in the profiler's own timeline"""

    def __init__(self):
        super().__init__(cuda=False)
        self.on = True

    def __call__(self, name):
        from torch.profiler import record_function
        with record_function("mk:" + name):
            pass


def count_by_section(events):
    """events: (name, start us, is_marker) of one profiled forward -> {segment: {runtime call: count}}; a segment is the stretch between two consecutive markers"""
    marks = sorted((t, n[3:]) for n, t, is_m in events if is_m)
    out = {}
    for n, t, is_m in events:
        if is_m or n not in LAUNCH_NAMES:
            continue
        k = max((i for i, (mt, _) in enumerate(marks) if mt <= t), default=None)
        if k is None or k + 1 >= len(marks):
            continue
        seg = marks[k][1] + " > " + marks[k + 1][1]
        out.setdefault(seg, {})
        out[seg][n] = out[seg].get(n, 0) + 1
    return out


def kernels(a):
    import jends_eval as JE
    import lane_gt_data as LD
    from torch.profiler import ProfilerActivity, profile
    assert torch.cuda.is_available(), "no GPU"
    runs = a.runs.split(",")
    before = gpu_state()
    if ("error" in before or before["util_max"] > 5.0) and not a.force:
        sys.exit(f"STOP: the GPU is not idle ({before})")
    LD.QData.DEFAULT_JUNCTION = True
    ds = LD.QData("val_clean", map_mode="error", pixels=False)
    x = LD.collate([ds[len(ds) // 2]])["x"].to("cuda").float()
    res = {}
    for r_ in runs:
        m, _ = JE.load_model(r_, "cuda")
        mk = ProfMarks()
        mk.on = False
        hook(m.jmodel.pyr, "pyr", mk)
        hook(m.jmodel.jm, "jm", mk)
        if m.nodes is not None:
            hook(m.nodes, "nodes", mk)
        hook(m.je, "je", mk)
        instrument(m.je, mk)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(5):
                m(x, gt=None)
            torch.cuda.synchronize()
            mk.on = True
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                mk("fwd:in")
                m(x, gt=None)
                mk("fwd:out")
                torch.cuda.synchronize()
        evs = [(e.name, float(e.time_range.start), e.name.startswith("mk:")) for e in prof.events()]
        res[r_] = count_by_section(evs)
    names = [BOUNDARIES[k][1] + " > " + BOUNDARIES[k + 1][1] for k in range(len(SECTION_NAMES))]
    stages = ["pyr:in > pyr:out", "jm:in > jm:out", "nodes:in > nodes:out"]
    print(f"CUDA runtime calls per stage / section of ONE forward (launches / memcopies / syncs), after {before}")
    print("  " + "segment".ljust(40) + "".join(r.rjust(22) for r in runs))
    allsegs = []
    for r_ in runs:
        for k in res[r_]:
            if k not in allsegs:
                allsegs.append(k)
    for k in allsegs:
        row = "  " + k.ljust(40)
        for r_ in runs:
            c = res[r_].get(k, {})
            row += (f"{c.get('cudaLaunchKernel', 0) + c.get('cudaLaunchKernelExC', 0)} / {c.get('cudaMemcpyAsync', 0) + c.get('cudaMemcpy', 0)} / {c.get('cudaStreamSynchronize', 0) + c.get('cudaDeviceSynchronize', 0)}" if c else "-").rjust(22)
        print(row)
    tot = {r_: {n: sum(c.get(n, 0) for c in res[r_].values()) for n in LAUNCH_NAMES} for r_ in runs}
    print("  " + "TOTAL".ljust(40) + "".join((f"{t['cudaLaunchKernel'] + t['cudaLaunchKernelExC']} / {t['cudaMemcpyAsync'] + t['cudaMemcpy']} / {t['cudaStreamSynchronize'] + t['cudaDeviceSynchronize']}").rjust(22) for t in tot.values()))
    out = a.out or os.path.join(ROOT, "runs", "eval", "jtiming", f"{a.tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    assert not os.path.exists(out), f"{out} exists"
    json.dump(dict(runs=runs, gpu_before=before, segments=res, totals=tot, frame="the middle VAL frame, bf16, batch 1"), open(out, "w"), indent=1)
    print("written", out)


def run(a):
''')
rep('    sub.add_parser("selftest")\n', '    k = sub.add_parser("kernels")\n    k.add_argument("--runs", default="e312_p_s0,e341_p_s0,e324_s_s0,e348_w_s0")\n    k.add_argument("--tag", default="e351_kernels")\n    k.add_argument("--out", default=None)\n    k.add_argument("--force", action="store_true")\n    sub.add_parser("selftest")\n')
rep("    run(a)\n\n\nif __name__", "    if a.cmd == \"kernels\":\n        return kernels(a)\n    run(a)\n\n\nif __name__")
# selftest: the segment counter
rep('    print("jtiming selftest ok")', '''    ev = [("mk:a", 0.0, True), ("cudaLaunchKernel", 1.0, False), ("cudaLaunchKernel", 2.0, False), ("mk:b", 3.0, True), ("cudaMemcpyAsync", 4.0, False), ("cudaStreamSynchronize", 5.0, False), ("mk:c", 6.0, True), ("cudaLaunchKernel", 7.0, False)]
    c = count_by_section(ev)
    assert c == {"a > b": {"cudaLaunchKernel": 2}, "b > c": {"cudaMemcpyAsync": 1, "cudaStreamSynchronize": 1}}, c     # a call after the last marker belongs to no segment
    print("jtiming selftest ok")''')
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
