"""Scratch: the toy test asserts what the critic found unasserted (warnings of predict, a refused run stores nothing and runs no frame, a short look keeps the health warning, plan finds what predict refuses)"""
p = "/mnt/c/Auto-research/bev_fusion/tests/test_runner_class.py"
s = open(p, encoding="utf-8", newline="").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)


rep("""    raises(r.predict, "unknown taps", "runs/thr0.7", "scenes:demo-0:val", taps=("nothing",), progress=0)
""", """    calls = TOY.STATE.calls
    raises(r.predict, "unknown taps", "runs/thr0.7", "scenes:demo-0:val", taps=("nothing",), progress=0)
    assert TOY.STATE.calls == calls and r.list_results("runs/thr0.7") == [], "a refused run runs no frame and leaves no (empty) result set behind"
    bad = r.plan("runs/thr0.7", "scenes:demo-0:val", taps=("nothing",))
    assert not bad["ok"] and "unknown taps" in bad["problems"][0] and TOY.STATE.calls == calls, "plan finds what predict would refuse"
""")
rep("""    print("  ok  tap: stored on request and read back; an unknown tap is refused before the network runs; time / probe say when the project has none")
""", """    print("  ok  tap: stored on request and read back; an unknown tap is refused before the network runs, stores nothing and plan says so; time / probe say when the project has none")

    # (4b) the health of the memory: predict returns the warning, a shorter look at the same scene does not erase it, a memory that is never filled by the network is no warning but is said
    try:
        TOY.STATE.memory = "flat"
        w = r.predict("runs/flat", "scenes:demo-1:val", progress=0)
        assert w["memory_checked"] is True and len(w["warnings"]) == 1 and "does not fill" in w["warnings"][0] and "demo-1" in w["warnings"][0], w
        look = r.predict("runs/flat", "frames:demo-1/0,demo-1/1:val", taps=("feat",), progress=0)
        assert look["frames_predicted"] == 2 and look["warnings"] == w["warnings"], ("a run of two frames cannot tell whether the memory fills: it keeps the recorded warning", look)
        again = r.predict("runs/flat", "scenes:demo-1:val", progress=0)
        assert again["predictions_reused"] is True and again["warnings"] == w["warnings"], again
        assert r.predict(net, "val", progress=0)["memory_checked"] is None, "nothing ran: nothing was checked"
    finally:
        TOY.STATE.memory = "ok"
    print("  ok  memory health: predict returns the warning of a memory that never fills; a shorter look at the scene keeps it; memory_checked says whether it could be judged")
""")
open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
