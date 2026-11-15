"""Scratch: stale statements and the project of the Evaluation in the tests (critic findings 11, 12)"""
T = "/mnt/c/Auto-research/bev_fusion/tests/"


def edit(name, pairs):
    s = open(T + name, encoding="utf-8", newline="").read()
    for old, new in pairs:
        assert s.count(old) == 1, (name, old[:80])
        s = s.replace(old, new)
    open(T + name, "w", encoding="utf-8", newline="").write(s)


edit("test_runner_class.py", [
    ("(the evaluation never imports the runner, the runner imports the evaluation's pieces)", "(the runner imports the evaluation's pieces; the evaluation reaches the runner only by the name in evalsuite.json)"),
    ("before the network runs, `time` / `probe` say when the project has none;", "before the network runs and plan says so too, a refused run stores nothing, the health of the memory (a warning that a shorter look keeps), `time` / `probe` say when the project has none;"),
])
edit("test_runner_real.py", [
    ("gives finite headline numbers (the run reads no ground truth, the scoring does)", "gives at least ten finite numbers and no loss (the run reads no ground truth, the scoring does)"),
    ('f"ev = Evaluation(); ev.resolve({net!r});', 'f"ev = Evaluation({ROOT!r}); ev.resolve({net!r});'),
    ("from runner.runner import Runner\\nfrom evalsuite.api import Evaluation\\n{code}", "from runner.runner import Runner\\nfrom evalsuite.api import Evaluation\\n{code}"),
])
print("ok")
