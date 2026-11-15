s = open("tests/test_student_window.py").read()


def cut(s, start, end):
    i, j = s.index(start), s.index(end)
    assert i < j, (start, end)
    return s[:i] + s[j:]


H = "# ------------------------------------------------------------------------------------------------ "
s = cut(s, 'print("A: OFF is the code before")', H + "B, C")
s = cut(s, "for name, v in model_checks().items():\n    check(v, name)\n# the real encoder", H + "D")
s = s.replace('for kind, tol in (("stub", TOL), ("resnet18", 1e-4)):\n    for name, v in stream_batch_checks(kind, tol).items():\n        check(v, name)\n', "")
for fn in ("jitter_fn_checks", "jitter_model_checks", "clip_checks"):
    s = s.replace(f"for name, v in {fn}().items():\n    check(v, name)\n", "")
old = "    for group in (data_checks, marker_checks, crash_checks, lock_checks, cli_checks):"
assert old in s
s = s.replace(old, "    for group in (marker_checks, crash_checks, lock_checks, cli_checks):")
open("temp/t26/dev_gh.py", "w").write(s.replace('os.path.dirname(os.path.abspath(__file__)), ".."', 'os.path.dirname(os.path.abspath(__file__)), "..", ".."'))
print(len(s))
