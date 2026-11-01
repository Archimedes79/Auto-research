"""Spec test of the TEST guard of evaluation/evaluate.py (review bug L2, decision D24): `test_allowed` accepts a run id only when the TEST usage log
names it as a whole id, the rule of ../autoresearch/tools/run.py (a prefix or a substring of a logged id must not pass); the card's protocol digest
and seed. No GPU, no data.
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


def test_digest_is_the_content_not_the_containers():
    import numpy as np
    a = {"pts": [np.arange(6.0).reshape(3, 2)], "id": 7, "z": (1, 2)}
    b = {"z": [1, 2], "id": 7, "pts": (np.arange(6.0).reshape(3, 2),)}                  # same content, other key order and container types
    assert EV.digest([a, a]) == EV.digest([b, b]) and len(EV.digest([a])) == 12
    moved = {"pts": [np.arange(6.0).reshape(3, 2) + 1e-9], "id": 7, "z": (1, 2)}
    assert EV.digest([a]) != EV.digest([moved]) and EV.digest([a]) != EV.digest([a, a]) and EV.digest([a]) != EV.digest([{**a, "id": 8}])
    assert EV.digest([[1], 2]) != EV.digest([[1, 2]]) and EV.digest({"a": "bc"}) != EV.digest({"ab": "c"}), "different structures never share a digest by accident"
    assert EV.digest(np.float64(1.5)) == EV.digest(1.5)
    x = np.arange(12.0).reshape(3, 4)
    assert EV.digest(x) != EV.digest(x.astype(np.float32)) and EV.digest(x) != EV.digest(x.reshape(4, 3)), "dtype and shape are part of the content"
    try:
        EV.digest([object()])
        raise AssertionError("an object without a stable content must be refused")
    except TypeError:
        pass


def test_seed_of_reads_the_run_card():
    assert EV.seed_of({"run": "e24_D7_j1rcs3"}) == 3                                    # a committed run card with "seed": 3
    assert EV.seed_of({"run": "no_such_run"}) is None and EV.seed_of({"kind": "graphs"}) is None
    assert EV.seed_of({"run": "e24_D7_j1rcs3", "seed": 5}) == 5, "the registry entry's own seed wins over the card"
    # the run card of this entry is a copy of another run's (seed 2): the entry says it is seed 0, so the card must not decide
    import json
    entry = json.load(open(f"{EV.R}/models/registry.json"))["e24_j1phd_s0_at50"]
    assert entry["seed"] == 0 and EV.seed_of(entry) == 0


def test_the_card_carries_protocol_and_seed():
    """evaluate() writes the digest of the ground truth it scored against and the seed of the run (stubs for the network and the metrics)"""
    import json
    import os
    import sys
    import tempfile
    import types
    import numpy as np
    gts = [{"pts": [np.arange(6, dtype=np.float32).reshape(3, 2)]}, {"pts": []}]
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(f"{tmp}/runs/e123_ladder/cards")
        json.dump({"run": "x", "seed": 3}, open(f"{tmp}/runs/e123_ladder/cards/xrun.json", "w"))
        fake = types.SimpleNamespace(card_any=lambda g, gt, extra=True: {"S": 0.5}, better=lambda m: {"S": "higher"})
        keep = (sys.modules.get("sparse_metrics"), EV.graphs_of, EV.R, EV.commit)
        sys.modules["sparse_metrics"], EV.R, EV.commit = fake, tmp, (lambda: "c")
        try:
            EV.graphs_of = lambda *a, **k: (["g", "g"], gts, {})
            card = EV.evaluate("m", {"m": {"kind": "builder", "run": "xrun"}}, "val")
            assert card["seed"] == 3 and card["protocol"] == {"gt_sha": EV.digest(gts)}, card
            other = EV.evaluate("m", {"m": {"kind": "graphs"}}, "val", tag="g")           # stored graphs: no run card, no seed
            assert "seed" not in other and other["protocol"] == card["protocol"], other
            moved = [{"pts": [np.arange(6, dtype=np.float32).reshape(3, 2) + 1]}, {"pts": []}]        # same frames and lines, another ground truth
            EV.graphs_of = lambda name, *a, **k: (["g", "g"], gts if name == "m" else moved, {})
            try:
                EV.evaluate("m", {"m": {"kind": "builder", "run": "xrun"}, "c": {"kind": "graphs"}}, "val", compare="c", tag="cmp")
                raise RuntimeError("a control scored against another ground truth must be refused")
            except AssertionError as e:
                assert "another ground truth" in str(e), e
        finally:
            if keep[0] is None:
                sys.modules.pop("sparse_metrics", None)
            else:
                sys.modules["sparse_metrics"] = keep[0]
            EV.graphs_of, EV.R, EV.commit = keep[1], keep[2], keep[3]


if __name__ == "__main__":
    n = 0
    for k, f in sorted(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            n += 1
    print(f"test_evaluate_guard: {n} tests passed")
