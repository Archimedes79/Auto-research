p = "bev_fusion/records/frame_changes.md"
s = open(p, encoding="utf-8").read()
marker = "## Changelog\n"
assert s.count(marker) == 1
line = ("- 2026-11-10 (clean-base round, Evaluation chat): **no number of any run changes** (`bev_selftest.py --golden`: the 3738 numbers of the former `evaluate_run.py` reproduced to 1e-9 on 24 real frames; "
        "`tests/test_evaluation_api.py --real` equals the engine). **The evaluator hash and the prediction identity change**, so a store or result set made before is not reused: `evaluation/bev_eval.py` joins "
        "`bev_adapter.NUMBER_PATHS` (the adapter hook `evaluator_extra()` and its place in `Evaluation._eval_id` are gone: one place for the code that shapes a number), and `bev_adapter.outputs_of` is the one function "
        "that builds a Subject's outputs (`BEVSubject.predict` and `bev_free.FreeSubject` both call it; it was a copy in `bev_free.py`). Consequences: `bev_suite.py eval` refuses to resume a store started before "
        "(`evaluator_conflict`), `Evaluation.show_numbers` warns 'scored with evaluator code X, now Y' on an evaluation made before, a new `Evaluation.run` predicts again under a new id. From now on an edit of "
        "`bev_eval.py` changes the hash as well.\n")
s = s.replace(marker, marker + line, 1)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
