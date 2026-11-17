import io
p = "evaluation/tests/test_jtest_eval.py"
s = io.open(p, encoding="utf-8", newline="").read()
def rep(a, b, cnt=1):
    global s
    assert s.count(a) == cnt, (a[:70], s.count(a))
    s = s.replace(a, b)
rep('code_drift=[], environment=ENV, device="cpu"), **over)', 'code_drift=[], environment=ENV, device="cpu", undigested_modules=[]), **over)')
rep('''    rc, *_ = run(card=rehearsal_card(files, environment=dict(ENV, files=dict(ENV["files"], **{"poses_val.npz": "other"}))))
    assert isinstance(rc, SystemExit) and "files/poses_val.npz" in str(rc)
''', '''    rc, *_ = run(card=rehearsal_card(files, environment=dict(ENV, files=dict(ENV["files"], **{"poses_val.npz": "other"}))))
    assert isinstance(rc, SystemExit) and "files/poses_val.npz" in str(rc)
    assert raises(RuntimeError, run, card=rehearsal_card(files, environment={})), "an empty environment record cannot vouch for the environment"
    assert raises(RuntimeError, run, card=rehearsal_card(files, device="cuda")), "a GPU rehearsal does not pin a CPU TEST run (the branch that is run must be the one rehearsed)"
    assert raises(RuntimeError, run, card=rehearsal_card(files, undigested_modules=["evaluation/x.py"])), "a digest that missed code which ran cannot pin the run"
    no_key = rehearsal_card(files)
    del no_key["undigested_modules"]
    assert raises(RuntimeError, run, card=no_key), "a card without the undigested-modules record cannot vouch for the digest"
    assert raises(RuntimeError, run, card=rehearsal_card(files, device=None)), "a card without a device cannot pin"
''')
rep('''def test_the_test_line_is_fixed_by_the_card():''', '''def test_standalone_dumps_are_refused_on_test_even_with_every_gate_open():
    """`dump-b` / `dump-ends --split test` would make a TEST dump without the pin, the environment check and the control check: refused by `main` before the gate, whatever the id and the launcher say"""
    old = os.environ.get("AR_RUN_ID")
    os.environ["AR_RUN_ID"] = "E343.t"
    try:
        for argv in (["dump-b", "--split", "test", "--seed", "0", "--test-id", "E343.t"], ["dump-ends", "--split", "test", "--seed", "0", "--test-id", "E343.t", "--run", "e312_p_s0"]):
            with patched((sys, "argv", ["jtest_eval.py"] + argv), (J, "test_gate_open", lambda *x, **k: (True, "")), (J, "dump_b", boom), (J, "dump_ends", boom)):
                try:
                    J.main()
                    raise AssertionError(f"{argv}: a standalone TEST dump ran")
                except SystemExit as e:
                    assert "made by `all` only" in str(e), (argv, str(e))
    finally:
        os.environ.pop("AR_RUN_ID", None)
        if old is not None:
            os.environ["AR_RUN_ID"] = old


def test_the_test_line_is_fixed_by_the_card():''')
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
