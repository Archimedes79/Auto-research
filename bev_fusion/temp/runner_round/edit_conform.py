p = "/mnt/c/Auto-research/bev_fusion/tests/test_suites_conform.py"
s = open(p, encoding="utf-8").read()
def rep(old, new, count=1):
    global s
    assert s.count(old) >= 1, old
    s = s.replace(old, new)
rep('''"""Are the suites one piece? The command lines of trainsuite (Training), runsuite (Runner) and evalsuite (Evaluation) against the convention they share''',
    '''"""Are the suites one piece? The command lines of trainsuite (Training), runsuite (Runner, RunJobs) and evalsuite (Evaluation) against the convention they share''')
rep('''only evalsuite / evaluation / torch / the run's `interface` module / the standard library (nothing of the trainer). Printed: the commands per class.''',
    '''only evalsuite / evaluation / torch / the run's `interface` module / the standard library (nothing of the trainer), runsuite in three layers that call only downwards (cli.py -> api.py ->
engine.py, jobs.py beside them) and the class Runner loads neither trainsuite nor the job layer. Printed: the commands per class.''')
rep('SUITES = {"trainsuite": "Training", "runsuite": "Runner", "evalsuite": "Evaluation"}', 'SUITES = {"trainsuite": ("Training",), "runsuite": ("Runner", "RunJobs"), "evalsuite": ("Evaluation",)}')
rep('''def imports_of(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            mods.add(n.module.split(".")[0])
    return mods
''', '''def imports_of(path, relative=False):
    """the top-level modules a file imports (relative=True: the sibling modules of its package it imports with `from .x import`)"""
    tree = ast.parse(open(path, encoding="utf-8").read())
    mods = set()
    for n in ast.walk(tree):
        if relative:
            if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module:
                mods.add(n.module.split(".")[0])
        elif isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            mods.add(n.module.split(".")[0])
    return mods
''')
rep('''    assert set(every) == {"Training", "Runner", "Evaluation"}, sorted(every)''', '''    assert set(every) == {c for cs in SUITES.values() for c in cs}, sorted(every)
    assert all({cls for (cls, _) in table[suite]} == set(SUITES[suite]) for suite in SUITES), "each suite offers exactly its classes"''')
rep('a class is offered by two suites (one Runner only, in runsuite)', 'a class is offered by two suites')
rep('["-m", "runsuite", "Runner.queue_list", "--json"]', '["-m", "runsuite", "RunJobs.queue_list", "--json"]')
rep('["-m", "runsuite", "Runner.status", "nope-xyz"]', '["-m", "runsuite", "RunJobs.status", "nope-xyz"]')
rep('''    print("commands per class:")
    for cls in ("Training", "Runner", "Evaluation"):''', '''    # runsuite: three layers that call only downwards, the job layer beside them
    rs = os.path.join(AR, "runsuite")
    layers = {"engine": set(), "api": {"engine"}, "jobs": set(), "cli": {"api", "jobs"}}
    for mod, allowed in layers.items():
        got = imports_of(os.path.join(rs, mod + ".py"), relative=True)
        assert got <= allowed, f"runsuite/{mod}.py imports {sorted(got - allowed)} (allowed: {sorted(allowed)})"
    assert imports_of(os.path.join(rs, "engine.py")) <= std | {"evalsuite", "numpy"}, imports_of(os.path.join(rs, "engine.py")) - std
    assert imports_of(os.path.join(rs, "api.py")) <= std | {"suitecli", "evalsuite"}, imports_of(os.path.join(rs, "api.py")) - std
    assert imports_of(os.path.join(rs, "cli.py")) <= std | {"suitecli"}, imports_of(os.path.join(rs, "cli.py")) - std
    rc, out, err = call("-c", "import sys, runsuite; runsuite.Runner('x'); print(sorted(m for m in sys.modules if m.split('.')[0] == 'trainsuite' or m in ('runsuite.jobs', 'runsuite.cli')))")
    assert rc == 0 and out.strip() == "[]", ("Runner loads the job layer or trainsuite", out, err[-200:])

    print("commands per class:")
    for cls in ("Training", "Runner", "RunJobs", "Evaluation"):''')
rep('commands in 3 suites, descriptions, verbs first, --json, exit codes 0/1/2 (also for an unknown id), layering', 'commands in 3 suites (4 classes), descriptions, verbs first, --json, exit codes 0/1/2 (also for an unknown id), layering (also the three layers of runsuite)')
open(p, "w", encoding="utf-8").write(s)
print("ok")
