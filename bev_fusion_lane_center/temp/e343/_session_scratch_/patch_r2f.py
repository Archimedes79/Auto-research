import os
p = os.path.join(os.environ["BFLC_ROOT"], "evaluation", "tests", "test_jtest_eval.py")
s = open(p, encoding="utf-8").read()
old = '''        assert raises(SystemExit, J.dump_b, "val", "e273_p_s0", p) and raises(SystemExit, J.dump_ends, "val", "e324_s_s0", p) and open(p, "rb").read() == b"x", "an existing dump is not overwritten"
        assert raises(RuntimeError, J.dump_b, "val", "e273_p_s0", os.path.join(d, "x.out")) and raises(RuntimeError, J.dump_ends, "val", "e324_s_s0", os.path.join(d, "y")), "no .pkl: the head file would take the name"
        assert sorted(os.listdir(d)) == ["x.pkl"]'''
new = '''        with patched((J, "frames_of", boom)):                                                          # a refusal comes before any work: reaching the frames fails the test (and never starts a dump)
            assert raises(SystemExit, J.dump_b, "val", "e273_p_s0", p) and raises(SystemExit, J.dump_ends, "val", "e324_s_s0", p) and open(p, "rb").read() == b"x", "an existing dump is not overwritten"
            assert raises(RuntimeError, J.dump_b, "val", "e273_p_s0", os.path.join(d, "x.out")) and raises(RuntimeError, J.dump_ends, "val", "e324_s_s0", os.path.join(d, "y")), "no .pkl: the head file would take the name"
        assert sorted(os.listdir(d)) == ["x.pkl"]'''
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new))
