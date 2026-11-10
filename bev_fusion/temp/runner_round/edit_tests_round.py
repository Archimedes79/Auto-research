"""Scratch: the test edits of the clean-base round (the Runner loses its unused methods() and its suitecli import)"""
T = "/mnt/c/Auto-research/bev_fusion/tests/"


def edit(name, pairs):
    s = open(T + name, encoding="utf-8", newline="").read()
    for old, new in pairs:
        assert s.count(old) == 1, (name, old[:70])
        s = s.replace(old, new)
    open(T + name, "w", encoding="utf-8", newline="").write(s)


edit("test_runner_class.py", [
    ("from evalsuite import selftest_api as TOY  # noqa: E402\n", "import suitecli  # noqa: E402\nfrom evalsuite import selftest_api as TOY  # noqa: E402\n"),
    ("in ('torch', 'trainsuite', 'nuscenes', 'cv2')))\" % ROOT)", "in ('torch', 'trainsuite', 'suitecli', 'nuscenes', 'cv2')))\" % ROOT)"),
    ("    infos = Runner.methods()\n", "    infos = suitecli.method_list(Runner)\n"),
    ('and not n.startswith("_") and n != "methods"]', 'and not n.startswith("_")]'),
])
edit("test_suites_conform.py", [
    ("for (cls, method), (word, doc) in cmds.items():", "for (cls, method), (_, doc) in cmds.items():"),
    ('assert runner <= std | {"numpy", "suitecli", "evalsuite"}, runner - std', 'assert runner <= std | {"numpy", "evalsuite"}, runner - std'),
    ("runner/runner.py only the standard library,\nnumpy, suitecli and evalsuite,", "runner/runner.py only the standard library,\nnumpy and evalsuite,"),
    ("the class: the evaluation's pieces and numpy, no training, no torch", "the class: the evaluation's pieces and numpy, no training, no torch, not even the command-line generator"),
])
print("ok")
