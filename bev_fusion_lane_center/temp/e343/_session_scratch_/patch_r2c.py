"""scratch (E343 code step, round 2c): parent packages in code_sha256, clean-ups of the new tests"""
import os

R = os.environ["BFLC_ROOT"]


def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (s.count(old), old[:80])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch(os.path.join(R, "evaluation", "jtest_eval.py"), [(
    '''            for d in roots:
                base = os.path.join(d, *name.split("."))
                hit = next((c for c in (base + ".py", os.path.join(base, "__init__.py")) if os.path.isfile(c)), None)
                if hit:
                    todo.append(os.path.realpath(hit))
                    break''', '''            parts = name.split(".")
            for d in roots:
                base = os.path.join(d, *parts)
                hit = next((c for c in (base + ".py", os.path.join(base, "__init__.py")) if os.path.isfile(c)), None)
                if hit:
                    todo += [os.path.realpath(hit)] + [os.path.realpath(c) for c in (os.path.join(d, *parts[:k], "__init__.py") for k in range(1, len(parts))) if os.path.isfile(c)]          # the packages above it run too
                    break''')])

patch(os.path.join(R, "evaluation", "tests", "test_jtest_eval.py"), [
    ('''    assert not raises(AssertionError, J.validate_dump, full[:-1], "val", "x") or True
''', ''),
    ('''        assert raises(SystemExit, J.precheck, ns(split="test", control_card=card, b_dump=None), setup, 0, True) is True or True
''', '''        with patched((J, "dump_dir", lambda split, n=0: d)):
            assert raises(SystemExit, J.precheck, ns(split="test", control_card=card), setup, 0, True), "TEST without --b-dump and without a dump at the default path: B's dump is never made on TEST"
'''),
    ('''        assert J.reading_inputs(cards[1])["lo"] == J.reading_inputs(cards[1])["hi"] and abs(J.reading_inputs(cards[1])["D"] - 0.008) < 1e-12''',
     '''        assert abs(J.reading_inputs(cards[1])["lo"] - J.reading_inputs(cards[1])["hi"]) < 1e-9 and abs(J.reading_inputs(cards[1])["D"] - 0.008) < 1e-12'''),
    ('''    with tempfile.TemporaryDirectory() as d, patched((J, "precheck", boom)):                           # a failed precheck: refused before the frames are read and before anything is scored
        import qgraph_eval as QE
        with patched((J, "frames_of", boom), (J, "score_seed", boom), (J, "guard_e321", lambda w: None), (J, "out_dir", lambda st, split, n=0: d), (J, "precheck", lambda *x: sys.exit("precheck refused")), (QE, "graph_numbers", toy_numbers)):
            assert raises(SystemExit, J.cmd_all, a)''', '''    with tempfile.TemporaryDirectory() as d:                                                           # a failed precheck: refused before the frames are read and before anything is scored
        with patched((J, "frames_of", boom), (J, "score_seed", boom), (J, "guard_e321", lambda w: None), (J, "out_dir", lambda st, split, n=0: d), (J, "precheck", lambda *x: sys.exit("precheck refused"))):
            assert raises(SystemExit, J.cmd_all, a)'''),
])
print("ok")
