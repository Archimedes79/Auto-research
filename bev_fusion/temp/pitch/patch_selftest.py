p = "evaluation/bev_selftest.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''    store = SimpleNamespace(frames=[{"key": k} for k in keys])
    a, e = SimpleNamespace(run=run, split="val"''', '''    store = SimpleNamespace(frames=[{"key": k} for k in keys], scene_names=lambda: ["scene-0001"])
    a, e = SimpleNamespace(run=run, split="val"''', 1)
s = s.replace('''    assert pv["data"]["n"] == 2 and pv["data"]["split"] == "val"
''', '''    assert pv["data"]["n"] == 2 and pv["data"]["split"] == "val"
    use = pv["data"]["use"]                                                    # which data was used, readable without code (2026-11-01)
    assert use["scene_names"] == ["scene-0001"] and use["scenes_evaluated"] == 1 and use["frames_evaluated"] == 2 and "nuScenes official split 'val'" in use["split_rule"], use
''', 1)
open(p, "w", encoding="utf-8").write(s)
