p = "tests/test_student_window.py"
s = open(p).read()
old = '''    import toy_train as TT
    ok = {}
'''
assert s.count(old) == 1
s = s.replace(old, '''    import toy_train as TT
    ok, out_root = {}, tempfile.mkdtemp(prefix="tj_", dir=tmp)                    # a fresh folder per call: a finished run of the same name would be resumed to its end and draw nothing
''')
old2 = "name=name, out_root=tmp, steps=8, accum=2"
assert s.count(old2) == 1
s = s.replace(old2, "name=name, out_root=out_root, steps=8, accum=2")
open(p, "w").write(s)
