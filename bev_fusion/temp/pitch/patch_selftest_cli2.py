def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:110])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/selftest_api.py", [
    ("from .api import Evaluation, EvaluationError, Frames\n", "from .api import Evaluation, EvaluationError, Frames\nfrom .runner_api import Runner\n"),
    ('''    assert code == 0 and "Evaluation.run(" in text and "Evaluation.compare(" in text and "gt_key=None" in text''', '''    assert code == 0 and "Evaluation.run(" in text and "Evaluation.compare(" in text and "Runner.run(" in text and "Runner.tap(" in text and "gt_key: 'Optional[str]' = None" in text'''),
    ('''    assert {m for m in Evaluation.METHODS if m in old} <= set(LEGACY), (old, Evaluation.METHODS)''', '''    assert {m for m in Evaluation.METHODS + Runner.METHODS if m in old} <= set(LEGACY), (old, Evaluation.METHODS, Runner.METHODS)
    code, text, _ = call("Runner.run", "runs/thr0.5", "val", "--json", "--workers", "0", "--progress", "0", *base)                 # the Runner alone: predictions without a ground truth
    ran = json.loads(text)
    assert code == 0 and ran["results"] == info["results"] and ran["frames_predicted"] == 0 and ran["predictions_reused"] is True, (code, text[:200])
    code, text, _ = call("tap", info["results"], "demo-0/2", *base)                                                                # a bare `tap` is the Runner's (the Evaluation has none)
    assert code == 0 and "taps" in text, text
    assert call("Runner.list", "runs/thr0.5", *base)[0] == 0 and call("Runner.check", "runs/thr0.5", "val", *base)[0] == 0'''),
])
print("ok")
