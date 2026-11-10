"""Scratch: adapt autoresearch/runsuite/selftest.py to the Runner / RunJobs split (one-off)."""
p = "/mnt/c/Auto-research/autoresearch/runsuite/selftest.py"
s = open(p, encoding="utf-8").read()
a = s.index('"""Self-test'), s.index('import contextlib')
s = s[:a[0]] + '''"""Self-test of runsuite on a toy project (a few seconds, no data, no GPU, a real run.py launcher): `python -m runsuite.selftest`.

The job layer (RunJobs: check / run / status / queue_add of kind "run", the modes check / precompute / network / full) is tested in trainsuite/selftest_api.py next to the queue it shares. Here:
the layers (the class Runner loads neither the job layer nor trainsuite nor the command line); the methods of Runner are the commands; the run without ground truth on the toy network of
evalsuite.selftest_api (plan finds the problems, predict stores, a second predict takes the stored frames, list_results and tap read them, time / probe say when the project has none, the
command line prints what the methods return); and the job mode "predict" of RunJobs (the frames text is checked before anything starts, the command is the project's `predict` command with the
run, the frames and the flags, a real run.py launch passes them on, the queue takes the job, a project without a `predict` command refuses)."""
''' + s[a[1]:]
s = s.replace("import contextlib\nimport io\nimport json\nimport os\nimport tempfile\n", "import contextlib\nimport io\nimport json\nimport os\nimport subprocess\nimport sys\nimport tempfile\n")
s = s.replace("from .api import Runner\n", "from .api import Runner\nfrom .jobs import RunJobs\n")
s = s.replace("rc = runsuite.CLI.dispatch(list(argv))", "rc = runsuite.dispatch(list(argv))")
i0 = s.index("    # the methods are the commands (no second list)")
i1 = s.index("    # the direct run, on the toy network of evalsuite")
new = '''    # the layers: a program that only runs a network does not load the job layer, trainsuite or the command line
    code = ("import sys, runsuite; r = runsuite.Runner('x'); "
            "print(sorted(m for m in sys.modules if m.split('.')[0] == 'trainsuite' or m in ('runsuite.jobs', 'runsuite.cli')))")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert p.returncode == 0 and p.stdout.strip() == "[]", (p.stdout, p.stderr[-300:])
    ok("layers: importing Runner loads neither the job layer, nor trainsuite, nor the command line")

    # the methods are the commands (no second list): every one has a whole-sentence summary; the helpers of the Evaluation are not commands
    names = [m["name"] for m in Runner.methods()]
    public = [n for n, f in vars(Runner).items() if callable(f) and not n.startswith("_") and n != "methods"]
    assert names == list(Runner.METHODS) == ["predict", "plan", "list_results", "tap", "time", "probe"] and set(public) == set(names) | {"resolve", "stored"}, (names, public)
    assert all(not m["summary"].endswith(("e.g.", "i.e.")) and m["summary"] for m in Runner.methods() + RunJobs.methods()), "a summary is a whole sentence"
    assert not set(Runner.METHODS) & set(RunJobs.METHODS), "a short command names one method: the two classes share no method name"
    ok(f"{len(names)} methods = {len(names)} commands, each with a summary; the two classes share no command word")

'''
s = s[:i0] + new + s[i1:]
old = 'assert rc == 2 and err.startswith("error:") and "Traceback" not in err, (rc, err)\n'
assert old in s
s = s.replace(old, old + '''    assert cli("Runner.nothing")[0] == 2 and cli("RunJobs.nothing")[0] == 2 and cli("nothing")[0] is None and cli("methods")[0] == 0, "an unknown command of a named class is an error"
    listing = cli("methods")[1]
    assert listing.count("Runner.") >= 6 and "RunJobs.check" in listing or "check  (RunJobs.check" in listing, "methods lists both classes"
''')
s = s.replace("pr = Runner(proj)", "pr = RunJobs(proj)")
s = s.replace("raises(Runner(proj).check, 'no \"runsuite\" block with \"predict\"'", "raises(RunJobs(proj).check, 'no \"runsuite\" block with \"predict\"'")
s = s.replace("job mode predict: the frames text", "job mode predict (RunJobs): the frames text")
open(p, "w", encoding="utf-8").write(s)
print("edited")
