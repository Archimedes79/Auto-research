def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:120])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/selftest_api.py", [
    ('''ME = sys.modules[__name__]
''', '''def evaluator_now(): return STATE.ev                                                # the evaluator code as it is now (what show_numbers compares a stored evaluation with)


class ToyRunner(Runner):
    """a project's own Runner, named in its evalsuite.json"""


ME = sys.modules[__name__]
'''),
    ('''    # the identity of the predictions: a checkpoint that changed, or another control, is another ResultSet (never the stored frames of the old one)''', '''    # the project names its Runner (evalsuite.json "runner"); an adapter handed in gets the generic one; a bad key is a message
    proj = tempfile.mkdtemp()
    os.makedirs(os.path.join(proj, "evaluation"))

    def config(runner):
        with open(os.path.join(proj, "evaluation", "evalsuite.json"), "w") as f:
            json.dump({"adapter": __name__, **({"runner": runner} if runner else {})}, f)
    config(f"{__name__}:ToyRunner")
    assert type(Evaluation(project=proj).runner) is ToyRunner and type(Evaluation(project=proj, adapter=ME).runner) is Runner, "the key is read when the adapter is the project's too"
    config(None)
    assert type(Evaluation(project=proj).runner) is Runner, "no key: the generic Runner"
    for bad, text in (("nonsense", "module:Class"), (f"{__name__}:Nothing", "cannot be built"), ("no_such_module_here:Runner", "cannot be built")):
        config(bad)
        _raises(EvaluationError, text, Evaluation, proj)
    shutil.rmtree(proj, ignore_errors=True)
    # the identity of the predictions: a checkpoint that changed, or another control, is another ResultSet (never the stored frames of the old one)'''),
    ('''    assert any("different evaluator code" in w for w in ev.compare(allr["id"], low_y["id"]).get("warnings", [])), "compare names a difference of the evaluator"''', '''    assert any("different evaluator code" in w for w in ev.compare(allr["id"], low_y["id"]).get("warnings", [])), "compare names a difference of the evaluator"
    assert any("scored with evaluator code" in w for w in ev.show_numbers(allr["id"])["warnings"]), "show_numbers says the counts are reduced with today's metrics"
    sub_c = ev.compare(allr["id"], sub["id"])
    assert sub_c["n_scenes"] == 1 and sub_c["only_a"] == 12 and sub_c["n_common_frames"] == 6, sub_c
    assert any("share 6 frames only" in w for w in sub_c["warnings"]) and any("only 1 scenes" in w for w in sub_c["warnings"]), "an interval on a subset and on few scenes says so"'''),
])
print("ok")
