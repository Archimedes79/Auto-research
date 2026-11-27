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
read-out, not the network's time (16.21 ms per frame, v1). The anchor re-scores the stored dump; it does not run the network again. The evaluator files (`adapter.py` `EVALUATOR_FILES`: `qgraph_eval.py`,
`jtoken_eval.py`, `sparse_lanes/` scorers, `lane_task/measures.py`) are hashed into every result: a change, even of a comment, makes old
runs incomparable (the judge refuses a `from` control), so they were left byte for byte; known stale text there (`_py14.sh` usage lines,
`evaluate.py` / `decide.py` mentions) and the timing's idle check `jtoken_eval.gpu_free` (it greps the gone `run.py`, so it never sees a
research job) go with the next real change of the evaluator, which then re-measures the champion. A TEST
evaluation needs the token cache of the test split first (`net/prep/lane_tokcache.py build --split test`; the test rows are in the val
features). Owner of the code: the lead chat of a round.
