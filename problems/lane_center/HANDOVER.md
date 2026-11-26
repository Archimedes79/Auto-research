# HANDOVER - lane_center (rewritten, not appended)

**State 2026-11-27.** The problem runs in the research system (restructure D154 / D155). The champion E372 S4 (D120) is reproduced bit for bit
through the new chain: the step `predict` (live dump, B' at batch 4) and the evaluation step (`evaluation/adapter.py`) give VAL topo F1
0.668172321247442 for seed 0, the number of its v1 card; junction F1 0.2538, F1@1.0 0.8459, lanes found 0.7232, Q 0.6796, vert100_dp 23.81.

**Next.**
1. The measurement E396 (the champion's two seeds through `research start`) makes `champions/lane_center/` (`research register`).
2. New work starts from the champion's settings in `training/configs/base.json`; a change enters the code default-off, the card switches it on.
3. Open since v1 (`PROBLEM.md` "Known gaps"): the SD-map association as a stage of the delivered chain, the past at token level, speed.

**Know.** VAL has 22 scenes: the judge warns that a scene bootstrap below about 30 scenes is narrow. A TEST evaluation needs the token cache of
the test split first (`net/prep/lane_tokcache.py build --split test`). The stages before B' (E271, E273) cannot be retrained bit for bit
(their code is lost since 2026-11-10, E273 used lane ground truth l3): B' is a kept input. Owner of the code: the lead chat of a round.
