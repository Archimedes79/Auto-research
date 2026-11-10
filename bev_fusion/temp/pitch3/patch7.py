def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:100])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8", newline="\n").write(s)
patch("autoresearch/evalsuite/suite.py", [
    ('"""EvalSuite: the one object the rest of the toolchain uses.\n',
     '"""EvalSuite: the engine\'s workbench, the object behind `Evaluation` (api.py, the front door of a person or an agent builds one for each call). Used directly by bev_suite.py, the lane project\n    and the engine\'s selftests (the older entry point), and the only way to work with a network that has no project adapter.\n'),
])
print("ok")
