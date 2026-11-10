def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:110])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/selftest_api.py", [
    ('''and "Runner.run(" in text and "Runner.tap(" in text and "gt_key=None" in text''', '''and "Evaluation.tap(" in text and "gt_key=None" in text and "Runner" not in text, "the Evaluation is the one front door of the command line"'''),
    ('''    assert {m for m in Evaluation.METHODS + Runner.METHODS if m in old} <= set(LEGACY), (old, Evaluation.METHODS, Runner.METHODS)
    code, text, _ = call("Runner.run", "runs/thr0.5", "val", "--json", "--workers", "0", "--progress", "0", *base)                 # the Runner alone: predictions without a ground truth
    ran = json.loads(text)
    assert code == 0 and ran["results"] == info["results"] and ran["frames_predicted"] == 0 and ran["predictions_reused"] is True, (code, text[:200])
    assert call("tap", info["results"], "demo-0/2", *base)[0] is None, "a bare word is the Evaluation's: the Runner's words need the class name"
    code, text, _ = call("Runner.tap", info["results"], "demo-0/2", *base)
    assert code == 0 and "taps" in text, text
    assert call("Runner.list", "runs/thr0.5", *base)[0] == 0 and call("Runner.check", "runs/thr0.5", "val", *base)[0] == 0
''', '''    assert {m for m in Evaluation.METHODS if m in old} <= set(LEGACY), (old, Evaluation.METHODS)
    # the layers: the command line offers the Evaluation only; what executes the network (tap, time, probe) is passed to the Runner
    assert call("Runner.run", "runs/thr0.5", "val", *base)[0] is None, "the Runner is not on this command line (runsuite has it)"
    code, text, _ = call("tap", info["results"], "demo-0/2", *base)
    assert code == 0 and "taps" in text, text
    for word, what in (("time", "no timing"), ("probe", "no probe")):
        code, _, err = call("Evaluation." + word, "runs/thr0.5", *base)
        assert code == 2 and what in err, (word, code, err)
    code, text, _ = call("Evaluation.list", "runs/thr0.5", *base)
    assert code == 0 and "predictions" in text
'''),
])
print("ok")
