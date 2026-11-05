p = "evaluation/bev_timing.py"
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new, 1)


rep('''BUSY_PATTERNS = ("train.py", "bev_suite.py", "evaluate_run", "run_ar.sh")     # a process whose command line has one of these may be using the GPU''',
    '''BUSY_SCRIPTS = ("train.py", "bev_suite.py", "evaluate_run.py")             # a PYTHON process running one of these scripts holds the GPU (the trainer, an evaluation)''')
a = s.index("def busy_processes(ps_text, own):")
b = s.index("def gpu_state(")
s = s[:a] + '''def holds_gpu(args):
    """does this command line (`ps -eo args`) belong to a process that holds the GPU? Only a python interpreter RUNNING a trainer / evaluation script does: a waiting
    run.py launcher (queued for a GPU slot, its own arguments name train.py and run_ar.sh), a bash wrapper and a shell are not users of the GPU. bev_suite on the CPU
    (--device cpu, or `modules` without --device) is not either."""
    tok = args.split()
    if not tok or not os.path.basename(tok[0]).startswith("python"):
        return False
    rest = [t for t in tok[1:] if not t.startswith("-")]
    if any(t in ("-c", "-m") for t in tok[1:3]) or not rest:                    # python -c / -m: not one of the scripts
        return False
    script = os.path.basename(rest[0])
    if script not in BUSY_SCRIPTS:
        return False
    if script == "bev_suite.py":
        on_cpu = "--device cpu" in args or "--device=cpu" in args or ("--device" not in args and " modules" in args)
        return not on_cpu
    return True


def busy_processes(ps_text, own):
    """the processes of a `ps -eo pid,args` listing that hold the GPU (holds_gpu), except the pids in `own`"""
    out = []
    for ln in ps_text.splitlines()[1:]:
        parts = ln.strip().split(None, 1)
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) not in own and holds_gpu(parts[1]):
            out.append(f"{parts[0]} {parts[1][:100]}")
    return out


''' + s[b:]
rep('''    ps = "  PID COMMAND\\n  101 /usr/bin/python -u runs/t16/code/train.py cfg\\n  102 python evaluation/bev_suite.py runs/x timing\\n  103 bash -c sleep 5\\n  104 python other.py\\n"
    assert [x.split()[0] for x in busy_processes(ps, {102})] == ["101"] and busy_processes(ps, {101, 102}) == [], "ps: trainers / evaluations, not ourselves"''',
    '''    ps = "  PID COMMAND\\n  101 /usr/bin/python -u runs/t16/code/train.py cfg\\n  102 python evaluation/bev_suite.py runs/x timing\\n  103 bash -c sleep 5\\n  104 python other.py\\n"
    assert [x.split()[0] for x in busy_processes(ps, {102})] == ["101"] and busy_processes(ps, {101, 102}) == [], "ps: trainers / evaluations, not ourselves"
    # queued jobs are not GPU users (the bug: a run.py launcher waiting for a slot, whose arguments name train.py / run_ar.sh / bev_suite, refused the timing on an idle GPU)
    waiting = ("  PID COMMAND\\n"
               "  201 python3 /mnt/c/Auto-research/autoresearch/tools/run.py --id t17.data --gpu --split train --hyp more scenes -- bash temp/train/run_ar.sh temp/compose/t17_data.json\\n"
               "  202 python3 ../autoresearch/tools/run.py --id t17.data.eval --gpu --split val --no-card --after t17.data -- bash temp/t16/eval_chain.sh runs/t17_data\\n"
               "  203 bash temp/train/run_ar.sh temp/compose/t17_data.json\\n"
               "  204 python3 ../autoresearch/tools/run.py --id x --gpu -- python evaluation/bev_suite.py runs/x eval --n 10\\n")
    assert busy_processes(waiting, set()) == [], busy_processes(waiting, set())
    running = ("  PID COMMAND\\n  301 /home/ml/miniconda3/envs/gcar/bin/python -u runs/t17_data/code/train.py temp/compose/t17_data.json\\n"
               "  302 python -u evaluation/bev_suite.py runs/t17_data eval --n 1000\\n  303 python -u evaluation/bev_suite.py runs/t17_data --device cpu eval --n 6\\n"
               "  304 python evaluation/bev_suite.py runs/t17_data modules --frames 3\\n  305 python evaluation/bev_suite.py runs/t17_data --device cuda modules\\n  306 python -c import train.py\\n")
    assert [x.split()[0] for x in busy_processes(running, set())] == ["301", "302", "305"], busy_processes(running, set())''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
