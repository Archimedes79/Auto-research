import os
R = os.environ["BFLC_ROOT"]
path = os.path.join(R, "evaluation", "tests", "test_jtest_eval.py")
s = open(path, encoding="utf-8").read()
pairs = [
    ('''if getattr(m, '__file__', None) and os.path.realpath(m.__file__).startswith(repo + os.sep)''',
     '''if os.path.isabs(getattr(m, '__file__', None) or '') and os.path.realpath(m.__file__).startswith(repo + os.sep)'''),
    ('''        a = ns(control_ends_dump=p, control_card=toy_card_file(d, 0.6, 0.5))
        assert J.precheck(a, setup, 0, False)[0]["ctl_ends"] == p, "the R1 line of the TEST command: E321's pair dump is not refused"''',
     '''        a = ns(control_ends_dump=p, control_card=toy_card_file(d, 0.6, 0.5))
        assert J.precheck(a, setup, 0, False)[0]["ctl_ends"] == p, "the R1 line of the TEST command: E321's pair dump is not refused (and the winner's B dump, still to be made, is not 'missing')"'''),
]
for old, new in pairs:
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new)
open(path, "w", encoding="utf-8").write(s)
print("ok")
