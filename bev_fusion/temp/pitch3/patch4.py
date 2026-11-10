def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:100])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8", newline="\n").write(s)
patch("bev_fusion/evaluation/README.md", [
    ("the outputs are built by the same functions as `BEVSubject.predict` builds them and are bit-equal to it; plugged into `Evaluation` through the adapter functions `free_source` / `free_subject` of `bev_eval.py`. The output-building lines are copied from the frozen `bev_adapter.py`: the test fails when they drift |",
     "the outputs are built by `bev_adapter.outputs_of`, the one function `BEVSubject.predict` uses too, and are bit-equal to it; plugged into `Evaluation` through the adapter functions `free_source` / `free_subject` of `bev_eval.py` |"),
])
print("ok")
