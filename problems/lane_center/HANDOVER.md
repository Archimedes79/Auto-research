# HANDOVER - lane_center (rewritten, not appended)

**State 2026-11-27.** The problem runs in the research system (restructure D154 / D155). E396 measured the champion E372 S4 (D120) through
`research start` from its snapshot: VAL topo F1 0.668172 / 0.681281 (seed 0 / 1), junction F1 0.2538 / 0.2481, Q 0.6796 / 0.6808 - the v1
numbers to the last digit. It is registered as the champion (`champions/lane_center/E396.champ/`). E397, the first comparison judged by the
system (keep threshold 0.5 against 0.6), is a TIE (-0.0026, [-0.0039, -0.0017]); the champion's decode stays.

**Next.**
1. New work starts from the champion's settings in `training/configs/base.json`: a change enters the code default-off with its test, the
   card switches it on; `research tests lane_center` and `research anchor lane_center E396.champ_s0` after a change of the code.
2. Open since v1 (`PROBLEM.md` "Known gaps"): the SD-map association as a stage of the delivered chain, the past at token level, speed.

**Know.** VAL has 22 scenes: the judge warns that a scene bootstrap below about 30 scenes is narrow. The steps `stage1` / `stage2` train
from the kept B' (E292 a; the stages before it, E271 and E273, cannot be retrained bit for bit: their code is lost since 2026-11-10). The
step `predict` dumps the delivered tokens (B' at batch 4), the evaluation step decodes and scores them: the `cost` in `result.json` is that
read-out, not the network's time (16.21 ms per frame, v1). The anchor re-scores the stored dump; it does not run the network again. The evaluator files (`adapter.py` `EVALUATOR_FILES`: every
lane module on the scored path stored dump -> prediction -> numbers; `evaluation/tests/test_evaluator_hash.py` keeps the list complete) are
hashed into every result: a change, even of a comment, makes old runs incomparable (the judge refuses a `from` control). The hash changed
with the interface round of 2026-11-28 (no number moved: the anchor agrees at tolerance 0), so the champion is measured again under it
before it serves as a control. A TEST
evaluation is not built yet: `net/prep/lane_tokcache.py build` takes train and val only and the token loaders refuse `test`, so
`research test lane_center` ends in the step `predict`; that path (the test rows are in the val features) is built before the owner's TEST
row. The same holds for BEV Fusion (`evaluation/bev_sets.py`). Owner of the code: the lead chat of a round.
