p = "evaluation/bev_suite.py"
s = open(p, encoding="utf-8").read()
def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new, 1)
rep('''    if a.command[0] == "generalisation":
        return cmd_generalisation(a, a.command[1:])
    from evalsuite.cli import main as suite_main''', '''    if a.command[0] == "generalisation":
        return cmd_generalisation(a, a.command[1:])
    if a.command[0] == "modules":                              # CPU unless --device is given: the probe must not disturb a training on the GPU
        from evaluation import bev_modules
        return bev_modules.main([a.run, "--device", a.device if "--device" in argv else "cpu", "--split", a.split] + (["--out", a.out] if a.out else []) + a.command[1:])
    if a.command[0] == "timing":
        from evaluation import bev_timing
        return bev_timing.main([a.run, "--device", a.device] + (["--out", a.out] if a.out else []) + a.command[1:])
    from evalsuite.cli import main as suite_main''')
rep('''    time                                        ms per frame of the chain (idle GPU only) -> eval/latency.json
''', '''    time                                        ms per frame of the chain (idle GPU only) -> eval/latency.json
    modules [--frames 3] [--threads 2]          module probe (evaluation/bev_modules.py; CPU unless --device is given): output statistics and dead channels of every
                                                module, the response when the camera / all radar detections / only the Doppler are removed, the velocity field on
                                                static and moving cells, a warm second frame for the memories, one PNG per frame -> eval/modules/
    timing [--n 30] [--force]                   ms per module at batch 1 + the forward / backward split of a training micro-batch (evaluation/bev_timing.py;
                                                idle GPU only, refused otherwise) -> eval/timing_modules.json
''')
open(p, "w", encoding="utf-8").write(s)
