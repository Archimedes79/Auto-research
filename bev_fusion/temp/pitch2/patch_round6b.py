def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:120])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("../bev_fusion/evaluation/bev_eval.py", [
    ('''def evaluator_extra() -> str:
    """the code that shapes the numbers outside the source's evaluator hash (bev_adapter.NUMBER_PATHS): this adapter (the reference, the metrics without the loss) and the label-free path;
    it goes into the id of an evaluation, so an edit of either scores the stored predictions again instead of serving old numbers"""
    return hashlib.sha1("".join(_file_hash(os.path.join(ROOT, "evaluation", f)) for f in ("bev_eval.py", "bev_free.py")).encode()).hexdigest()[:12]
''', '''def evaluator_extra() -> str:
    """the code that shapes the numbers outside the source's evaluator hash (bev_adapter.NUMBER_PATHS): this adapter (the reference, the metrics without the loss), the label-free path and
    bev_views.py / bev_annotation_pts.py (the masks that are every map's denominator, the boxes and the layer index: imported by the metrics but not in NUMBER_PATHS, a frozen file);
    it goes into the id of an evaluation, so an edit of any of them scores the stored predictions again instead of serving old numbers"""
    files = ("bev_eval.py", "bev_free.py", "bev_views.py", "bev_annotation_pts.py")
    return hashlib.sha1("".join(_file_hash(os.path.join(ROOT, "evaluation", f)) for f in files).encode()).hexdigest()[:12]


def evaluator_now():
    """the evaluator hash of the code as it is now (what a stored evaluation was scored with is in its record): `show_numbers` warns when they differ"""
    from evaluation import bev_adapter
    return bev_adapter.evaluator_hash()
'''),
])
patch("../bev_fusion/tests/test_evaluation_api.py", [
    ('''    assert cfg == {"adapter": "evaluation.bev_eval"}, cfg''', '''    assert cfg == {"adapter": "evaluation.bev_eval", "runner": "runner.runner:Runner"}, cfg'''),
    ('''    assert len(bev_eval.evaluator_extra()) == 12, "the adapter's own code is part of the evaluation id"''', '''    assert len(bev_eval.evaluator_extra()) == 12, "the adapter's own code is part of the evaluation id"
    ev_ = Evaluation(project=ROOT)                                                                      # the project names its Runner: bev_fusion/runner/runner.py, built with (project, adapter)
    assert type(ev_.runner).__module__ == "runner.runner", type(ev_.runner)
    assert type(Evaluation(adapter="evaluation.bev_eval").runner).__module__ == "evalsuite.runner_api", "an adapter handed in gets the generic Runner, not the project's"'''),
    ('''    from evalsuite.api import ADAPTER_FUNCTIONS
    missing =''', '''    from evalsuite.api import ADAPTER_FUNCTIONS, Evaluation
    missing ='''),
    ('''        ev = Evaluation(adapter="evaluation.bev_eval")
        # 1 the interface''', '''        ev = Evaluation(project=ROOT)                                                                     # the project's own Runner (runner/runner.py) predicts, the Evaluation scores
        assert type(ev.runner).__module__ == "runner.runner", type(ev.runner)
        # 1 the interface'''),
])
print("ok")
