import re


def rd(p):
    return open(p, encoding="utf-8").read()


def wr(p, s):
    open(p + ".new_s292", "w", encoding="utf-8", newline="\n").write(s)


def sub(s, a, b):
    assert s.count(a) == 1, a[:80]
    return s.replace(a, b)


q = rd("evaluation/qends_eval.py")
q = sub(q, "import jwire_head as JH  # noqa: E402\n", "")                                                      # torch is imported only when a head is asked for
i0 = q.index('    wire = getattr(a, "wire", None)\n    assert wire in (None, "twin", "head")')
i1 = q.index("    base_pt = a.base or os.path.join(ROOT")
q = q[:i0] + ('    wire, wire_run = check_wire(a)\n'
              '    if wire == "head":\n'
              '        import jwire_head as JH                                                                                         # E325: the learned head `runs/<run>/model.pt`, or `<run>:<file>` for another checkpoint of the run (model_h1.pt: a second head seed)\n'
              '        head = JH.load_wire(*wire_run.partition(":")[::2])\n') + q[i1:]
q = sub(q, "def cmd_score(a):\n", '''def check_wire(a):
    """-> (wire, wire_run) after the option checks of `--wire` / `--wire-run` (E325): twin / head need --snap and none of --attach, --attach-r, --snap-all, --snap-guard (the head was not trained on attached ends, the others
    override a choice); `--wire head` needs `--wire-run` and only it takes it"""
    wire, wire_run = getattr(a, "wire", None), getattr(a, "wire_run", None)
    assert wire in (None, "twin", "head"), f"--wire {wire!r}: twin or head"
    assert wire is None or (bool(getattr(a, "snap", False)) and not any(getattr(a, k, False) for k in ("snap_all", "snap_guard", "attach")) and getattr(a, "attach_r", None) is None), "--wire twin / head needs --snap and none of --attach, --attach-r, --snap-all, --snap-guard"
    assert (wire == "head") == bool(wire_run), "--wire head needs --wire-run <run of the head> (and --wire-run only goes with --wire head)"
    return wire, wire_run


def cmd_score(a):
''')
# the head branch inside the loop: JH is a local import now
q = sub(q, '        elif wire == "head":                                                                                                    # E325 arm w: the learned head chooses the pieces (its features never read the GT)\n',
        '        elif wire == "head":                                                                                                    # E325 arm w: the learned head chooses the pieces (its features never read the GT)\n')
# selftest
q = sub(q, '    print("qends_eval selftest ok")\n', '''    # E325: the option checks of --wire / --wire-run
    from types import SimpleNamespace as NS2
    for ok_opt in (NS2(), NS2(wire="twin", snap=True), NS2(wire="head", snap=True, wire_run="e325_w_s0")):
        check_wire(ok_opt)
    for bad_opt in (NS2(wire="twin"), NS2(wire="head", snap=True), NS2(wire="twin", snap=True, wire_run="x"), NS2(wire="head", snap=True, wire_run="x", attach=True), NS2(wire="head", snap=True, wire_run="x", attach_r=1.5),
                    NS2(wire="twin", snap=True, snap_all=True), NS2(wire="twin", snap=True, snap_guard=True), NS2(wire="soft", snap=True), NS2(snap=True, wire_run="x")):
        try:
            check_wire(bad_opt)
        except AssertionError:
            continue
        raise AssertionError(f"accepted {bad_opt}")
    print("qends_eval selftest ok")
''')
wr("evaluation/qends_eval.py", q)

# start_w.sh: _pyr.sh (retries on the CUDA unknown error)
s = rd("temp/e325/start_w.sh")
s = sub(s, "  -- bash tools/env/_py14.sh jwire_train.py run", "  -- bash tools/env/_pyr.sh jwire_train.py run")
open("temp/e325/start_w.sh", "w", encoding="utf-8", newline="\n").write(s)
print("ok")
