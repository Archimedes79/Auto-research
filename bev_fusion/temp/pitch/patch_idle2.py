p = "evaluation/bev_timing.py"
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new, 1)


rep('''BUSY_SCRIPTS = ("train.py", "bev_suite.py", "evaluate_run.py")             # a PYTHON process running one of these scripts holds the GPU (the trainer, an evaluation)''',
    '''GPU_SCRIPTS = {"train.py": "cuda", "train_motion_block.py": "cuda", "bev_suite.py": "cuda", "bev_runner.py": "cuda", "bev_timing.py": "cuda",
               "bev_prototype.py": "cuda", "cache_bevfusion_features.py": "cuda", "bev_modules.py": "cpu"}        # script -> the device it uses when --device is not given
SUITE_OPTIONS_WITH_VALUE = ("--device", "--workers", "--split", "--ckpt-dir", "--out", "--control", "--state")    # bev_suite.py's global options that take a value''')
a = s.index("def holds_gpu(args):")
b = s.index("def busy_processes(")
s = s[:a] + '''def holds_gpu(args):
    """does this command line (`ps -eo args`) belong to a process that holds the GPU? Only a python interpreter RUNNING one of GPU_SCRIPTS does (the script is the first
    token ending in .py, or the module of `-m`; `-W ignore` / `-X dev` values are not mistaken for it), on the device it was given (--device, else the script's default;
    bev_suite's `modules` command defaults to the CPU). A waiting run.py launcher (queued for a GPU slot, its own arguments name train.py and run_ar.sh), a bash wrapper,
    `python -c` and a shell are not users of the GPU."""
    tok = args.split()
    while tok and (os.path.basename(tok[0]) == "env" or "=" in tok[0]):         # `env VAR=1 python ...`
        tok = tok[1:]
    if not tok or not os.path.basename(tok[0]).startswith("python"):
        return False
    opts = tok[1:]
    if "-c" in opts[:4]:
        return False
    if "-m" in opts[:4]:
        i = opts.index("-m")
        script, after = opts[i + 1].split(".")[-1] + ".py" if i + 1 < len(opts) else "", opts[i + 2:]
    else:
        i = next((k for k, t in enumerate(opts) if t.endswith(".py")), None)
        script, after = (os.path.basename(opts[i]), opts[i + 1:]) if i is not None else ("", [])
    if script not in GPU_SCRIPTS:
        return False
    device = GPU_SCRIPTS[script]
    m = re.search(r"--device(?:=|\\s+)(\\S+)", args)
    if m:
        device = m.group(1)
    elif script == "bev_suite.py":
        positional, k = [], 0
        while k < len(after) and len(positional) < 2:
            if after[k].startswith("--"):
                k += 2 if after[k] in SUITE_OPTIONS_WITH_VALUE else 1
            else:
                positional.append(after[k])
                k += 1
        if len(positional) == 2 and positional[1] == "modules":
            device = "cpu"
    return device.startswith("cuda")


''' + s[b:]
rep("import json\nimport os\nimport statistics", "import json\nimport os\nimport re\nimport statistics")
rep("the other GPU processes (`nvidia-smi` shows none under WSL2, so `ps` is read too: any\ntrainer / evaluation process counts)",
    "the other GPU processes (`nvidia-smi` shows none under WSL2, so `ps` is read too: a python\nprocess running a trainer / evaluation / probe script on the GPU counts, a run.py launcher waiting for a slot does not)")
rep('''This is the fp32 path of `BEVSubject.benchmark()`''', '''A job that is merely QUEUED (a run.py launcher waiting for its GPU slot) is not seen, and when a slot frees it starts under a running timing: time through a GPU
slot (as `temp/t16/eval_chain.sh` does) or look at the schedule first.
This is the fp32 path of `BEVSubject.benchmark()`''')
# selftest: more cases
rep('''    assert [x.split()[0] for x in busy_processes(running, set())] == ["301", "302", "305"], busy_processes(running, set())''',
    '''    assert [x.split()[0] for x in busy_processes(running, set())] == ["301", "302", "305"], busy_processes(running, set())
    for line, want in (("python3.12 -u runs/x/code/train.py cfg.json", True), ("python -W ignore runs/x/code/train.py cfg.json", True), ("python -X dev -u train.py cfg", True),
                       ("env CUDA_VISIBLE_DEVICES=0 python train.py cfg", True), ("python -m evaluation.bev_suite runs/x eval --n 5", True),
                       ("python -m evaluation.bev_suite runs/x --device=cpu eval --n 5", False), ("python evaluation/bev_suite.py runs/x eval --n 5 --out modules", True),
                       ("python evaluation/bev_suite.py runs/x --workers 0 modules", False), ("python evaluation/bev_suite.py runs/modules eval", True),
                       ("python evaluation/bev_suite.py runs/x --curves --fusion eval", True), ("python evaluation/bev_suite.py runs/x --device=cuda modules", True),
                       ("python evaluation/bev_suite.py runs/x --device cuda modules", True), ("python runner/bev_runner.py full runs/x", True),
                       ("python runner/bev_runner.py full runs/x --device cpu", False), ("python evaluation/bev_timing.py runs/x", True), ("python evaluation/bev_modules.py runs/x", False),
                       ("python evaluation/bev_modules.py runs/x --device cuda", True), ("python evaluation/bev_prototype.py runs/x", True),
                       ("python -c train.py", False), ("python -m json.tool train.py", False), ("python3 tools/run.py --id x --gpu -- python train.py", False),
                       ("bash temp/train/run_ar.sh cfg train.py", False), ("python other.py train.py", False)):
        assert holds_gpu(line) == want, (line, want)''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
