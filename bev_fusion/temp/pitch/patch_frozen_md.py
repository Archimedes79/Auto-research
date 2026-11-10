p = "FROZEN.md"
s = open(p, encoding="utf-8").read()
marker = "## Changelog\n"
assert s.count(marker) == 1
line = ("- 2026-11-10: `evaluation/bev_suite.py` and `bev_adapter.py` (frozen) get named sets (Mirko 2026-11-10: one set definition, not several; the definition is the training's `splits/*.json`, "
        "read by the new `evaluation/bev_sets.py`): `--split` takes `<split>.<part>` (`small.val`, `large.train`) next to `val` / `train`; a named set writes files and stores of its own name, "
        "its scene hash is part of the store identity, TEST and reserved split names are refused, the commands that write the run's plain files refuse a named set. **No number of any old run "
        "changes**: for `val` / `train` every file name, store name, card key and store identity is as before (`describe()` gets `set_hash` only for a named set). Checked by "
        "`bev_selftest.py --golden` (3738 numbers of the former `evaluate_run.py` reproduced to 1e-9 on 24 real frames), `tests/test_evaluate_door.py --real` (door = suite, 5237 numbers) and `--sets` "
        "(on `t16_full_s0`: `large.val` / `large.train` give the numbers of `val` / `train` number for number, `small.val` evaluates exactly its 10 scenes), `bev_selftest.py` 17 checks. "
        "`ar-critic` pass done, 9 findings fixed (`--time` on a set, output folders of pictures, silent drop of a set's scenes, the set hash, reserved split names, tag collisions, the prior store). "
        "By hand: a new feature of the frame, not a bug fix, made on Mirko's order of this day; the evaluator hash changes (a store started before is not resumed, an old card pairs with a new one "
        "only with `allow: provenance`).\n")
open(p, "w", encoding="utf-8").write(s.replace(marker, marker + line))
print("ok")
