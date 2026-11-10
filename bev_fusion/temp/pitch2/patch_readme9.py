def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:120])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/README.md", [
    ("| `get_cost(id)`, `report(id)`, `make_card(id)` | ms per frame, the html page, the metric card in the project's contract (refused when the checkpoint or the evaluator code changed since, or the predictions are gone: score again) |",
     "| `get_cost(id)`, `report(id)`, `make_card(id, out)` | ms per frame, the html page, the metric card in the project's contract WRITTEN as a file (default `<net>/eval/card_<e-id>.json`, or `out`; -> `{path, card}`; a verdict reads it: name it in the experiment card's `arm.cards`, check it with `python autoresearch/tools/card.py check FILE`); refused when the checkpoint or the evaluator code changed since, or the predictions are gone: score again |"),
    ("`get-cost ID`, `report ID`, `make-card ID`, `draw-frame ID FRAME` |", "`get-cost ID`, `report ID`, `make-card ID [--out FILE]`, `draw-frame ID FRAME` |"),
])
patch("evalsuite/api.py", [
    ('''Intermediate results: `run(..., taps=("fused", ...))`''', '''What a source's `describe()` must give: `gt_signature` and `evaluator_hash` (the hash of the files that shape the numbers); without either the Evaluation refuses (a hash that cannot be computed is not "unchanged").
Intermediate results: `run(..., taps=("fused", ...))`'''),
])
patch("../bev_fusion/evaluation/README.md", [
    ('''ev.find_worst(r["id"], "seg"), ev.make_card(r["id"])''', '''ev.find_worst(r["id"], "seg"), ev.make_card(r["id"])   # make_card writes runs/<name>/eval/card_<e-id>.json (a verdict reads it: arm.cards in the experiment card)'''),
])
print("ok")
