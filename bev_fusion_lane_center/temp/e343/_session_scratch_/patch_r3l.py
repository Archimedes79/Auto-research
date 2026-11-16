import os
R = os.environ["BFLC_ROOT"]
path = os.path.join(R, "evaluation", "tests", "test_jtest_eval.py")
s = open(path, encoding="utf-8").read()
pairs = [
    ('''        with patched((J, "frames_of", boom)):                                                          # a refusal comes before any work: reaching the frames fails the test (and never starts a dump)
            assert raises(SystemExit, J.dump_b,''', '''        with patched((J, "frames_of", boom), (J, "EXTRA_WRITABLE", (d,))):                             # a refusal comes before any work: reaching the frames fails the test (and never starts a dump)
            assert raises(SystemExit, J.dump_b,'''),
    ('''        assert raises(PermissionError, J.dump_b, "val", "e273_p_s0", os.path.join(d, "z.pkl")) and raises(PermissionError, J.dump_ends, "val", "e324_s_s0", os.path.join(d, "z.pkl")), "a place outside the allow-list: refused at the START of the dump, not after it"''',
     '''        with patched((J, "frames_of", boom)):
            assert raises(PermissionError, J.dump_b, "val", "e273_p_s0", os.path.join(d, "z.pkl")) and raises(PermissionError, J.dump_ends, "val", "e324_s_s0", os.path.join(d, "z.pkl")), "a place outside the allow-list: refused at the START of the dump, not after it"'''),
    ('''    with tempfile.TemporaryDirectory() as d, patched((J, "provenance", lambda run, device: seen.append((run, device)) or PROV(run, device)), (J, "frames_of", stop)):''',
     '''    with tempfile.TemporaryDirectory() as d, patched((J, "provenance", lambda run, device: seen.append((run, device)) or PROV(run, device)), (J, "frames_of", stop), (J, "EXTRA_WRITABLE", (d,))):'''),
]
for old, new in pairs:
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new)
open(path, "w", encoding="utf-8").write(s)
print("ok")
