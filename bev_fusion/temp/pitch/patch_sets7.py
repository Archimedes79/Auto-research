def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:90])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evaluation/evaluate.py", [
    ('''        return f"{a.run} is not a run folder (no config.json); run from the bev_fusion folder"
''', '''        return f"{a.run} is not a run folder (no config.json); run from the bev_fusion folder"
    try:
        bev_sets.lists(a.split, [], [])                                             # the file of a named set exists (val / train need none)
    except ValueError as ex:
        return str(ex)
'''),
])

patch("tests/test_evaluate_door.py", [
    ('''    assert refused("--qual", "scene-0061") and refused("--videos", "scene-0061/12")''',
     '''    # sets (evaluation/bev_sets.py): a named set goes to bev_suite as it is; a malformed name and TEST are refused, --all is the run's own lists
    assert steps_of("--quant", "--split", "small.val")[0]["argv"][:6] == ["--device", "cuda", "--workers", "2", "--split", "small.val"]
    assert steps_of("--qual", "3", "--split", "large.train")[0]["argv"][4:6] == ["--split", "large.train"] and steps_of("--time", "--split", "xl.val")
    assert all(refused("--quant", "--split", x) for x in ("small.test", "small", "Small.val", "small.val.x", ".val", "")) and refused("--all", "--split", "small.val")
    assert refused("--qual", "scene-0061") and refused("--videos", "scene-0061/12")'''),
    ('''        for opts in (["--quant"], ["--time"], ["--all"], ["--videos", "2"], ["--qual", "3"], ["--quant", "--split", "train"]):''',
     '''        assert "no split 'nosuch'" in E.refusal(ns("--quant", "--split", "nosuch.val"), lister), "a set whose split file does not exist"
        for opts in (["--quant"], ["--time"], ["--all"], ["--videos", "2"], ["--qual", "3"], ["--quant", "--split", "train"], ["--quant", "--split", "small.val"]):'''),
    ('''                                      ("card_train.json", ["--quant", "--split", "train"], True), ("card.json", ["--videos", "2"], False),''',
     '''                                      ("card_train.json", ["--quant", "--split", "train"], True), ("card.json", ["--videos", "2"], False),
                                      ("card_small_val.json", ["--quant", "--split", "small.val"], True), ("card.json", ["--quant", "--split", "small.val"], False),
                                      ("card_small_val.json", ["--quant"], False), ("card_train.json", ["--quant", "--split", "small.train"], False),'''),
])
print("ok")
