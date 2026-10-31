"""Spec test of the TEST guard of evaluation/evaluate.py (review bug L2, decision D24): `test_allowed` accepts a run id only when the TEST usage log
names it as a whole id, the rule of ../autoresearch/tools/run.py (a prefix or a substring of a logged id must not pass). No GPU, no data.
Run (from the problem folder): wsl bash tools/env/_py14.sh test_evaluate_guard.py"""
import evaluate as EV

LOG = """| date | run id | ok |
|---|---|---|
| 2026-10-20 | E212.a0 | Mirko, in chat |
| 2026-10-21 | `E245.b1`, E245.b2 | grant of the loop prompt |
"""


def test_whole_ids_pass():
    for rid in ("E212.a0", "E245.b1", "E245.b2"):
        assert EV.test_allowed(rid, LOG), rid


def test_prefixes_and_substrings_fail():
    for rid in ("", "E", "E2", "E212", "E212.a", "212.a0", "E245.b", "a0", "E212.a01", "E245", "b1", "E245.b12"):
        assert not EV.test_allowed(rid, LOG), rid
    # tokens of the log that are not run ids (date parts, words) never pass: the id must have the shape E<number>...
    assert not EV.test_allowed("2026", LOG) and not EV.test_allowed("10", LOG) and not EV.test_allowed("Mirko", LOG)


def test_regex_characters_are_literal():
    assert not EV.test_allowed("E212.a.", LOG) and not EV.test_allowed("E2.2.a0", LOG) and not EV.test_allowed(".*", LOG)


def test_reads_the_project_log_by_default():
    assert EV.test_allowed("E", None) is False  # the real log is read and a one-letter id never passes


if __name__ == "__main__":
    n = 0
    for k, f in sorted(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            n += 1
    print(f"test_evaluate_guard: {n} tests passed")
