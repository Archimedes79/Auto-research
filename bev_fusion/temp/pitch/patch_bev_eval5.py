def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:110])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evaluation/bev_eval.py", [
    ('''_CODE = {"path": None}                           # the code snapshot of the first network built in this process: its modules stay imported, no other snapshot can run here
''', '''_CODE = {"path": None}                           # the code snapshot of the first network built in this process: its modules stay imported, no other snapshot can run here
_SOURCES = {}                                    # (run, set, workers, control) -> its Source while it holds its Subject: the Evaluation and the Runner ask for the same one in a row
'''),
    ('''    A = _A()
    src, subj, _ = A.build(net, device="cpu", split=of, workers=workers, control=control)
    src._bev_subject = subj
    return src''', '''    key = (net, of, workers, control)
    if key in _SOURCES and getattr(_SOURCES[key], "_bev_subject", None) is not None:
        return _SOURCES[key]                                            # built a moment ago for the same question: the model is loaded once
    A = _A()
    src, subj, _ = A.build(net, device="cpu", split=of, workers=workers, control=control)
    src._bev_subject = subj
    src._bev_key = key
    _SOURCES[key] = src
    return src'''),
    ('''def release(src) -> None:
    """drop the Subject a source holds (a check or a rescoring never runs it)"""
    src._bev_subject = None''', '''def release(src) -> None:
    """drop the Subject a source holds (a check or a rescoring never runs it)"""
    src._bev_subject = None
    _SOURCES.pop(getattr(src, "_bev_key", None), None)'''),
    ('''    return bev_suite._task_only(net, "curves" in metrics, "fusion" in metrics, "velocity" in metrics)''', '''    t = bev_suite._task_only(net, "curves" in metrics, "fusion" in metrics, "velocity" in metrics)
    metrics_of = t.metrics
    t.metrics = lambda: [m for m in metrics_of() if m.name != "loss"]       # the Runner reads no ground truth, so a prediction carries no validation loss (D72); `bev_suite.py eval` keeps it
    return t'''),
    ('''    gts = {fr["info"].get("gt_at_predict") for fr in store.frames if fr["info"].get("gt_at_predict")}
    if gts not in (set(), {info.get("gt_signature")}):                          # predicted under another GT: its val loss is not this GT's
        flat = {k: v for k, v in flat.items() if not k.startswith("val_loss/")}
        better = {k: v for k, v in better.items() if not k.startswith("val_loss/")}
''', ''),
])
print("ok")
