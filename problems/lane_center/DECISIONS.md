# Decisions of the lane-centre problem

The owner's decisions about this problem (permissions, orders, the champion). A question waits in the Open table until the owner answers; an agent
records the answer as a decided row and never acts before. Ids `D<n>` are counted together with the root `DECISIONS.md`. The rows before the
restructure of 2026-11-27 are in the tag `v1` (`bev_fusion_lane_center/experiment_execution/DECISIONS.md`); the rows that still hold are carried
here, the technical rules among them live in `DESIGN.md` (R1 - R7).

## Decided
| date | decision |
|---|---|
| 2026-11-04 | D55: the system is end-to-end differentiable: soft steps up to the output, a loss on the output reaches every module, matching inside the loss is allowed, hard selection only on the output and named. Now: `DESIGN.md` R1. |
| 2026-11-05 | D57: this problem has first call on the GPU, the queue and chat time; every other problem comes second. A running job of another problem is never stopped for it. |
| 2026-11-16 | D113: the mission is an end-to-end differentiable token champion that is good, simple and cheap; rule-step systems are never a bar. Now: `DESIGN.md` R3, R4. |
| 2026-11-18 | D120: the champion is the two-stage token recipe with the S4 stage 2 (E372 S4: stage 1 E373 cb on the frozen lane head B' = E292 a, stage 2 trained together). Now: `DESIGN.md` R6; `champions/lane_center/`. |
| 2026-11-26 | D154: the restructure into the research system (see the root `DECISIONS.md`); this folder holds the champion's code, moved from `bev_fusion_lane_center/` with its history. |

## Open
| id | question |
|---|---|
| Q-2026-11-28a | For the record, and the owner's word whether it needs a row in `TEST_LOG.md`: on 2026-11-28 about 01:35 a coder's first two runs of the new feature check (`net/prep/foundation/check_features.py`, commit `8b1e4fe2`) re-extracted the foundation features of 8 TEST frames (the part `val` of the stored features holds VAL 873 + TEST 833 rows) and compared them with the stored features. No ground truth was read, no lane network ran, no score was made; nothing was chosen on it. The tool reads TRAIN and VAL rows only since then (a test pins it), and the result file was written again from a run without TEST rows. |
| Q-2026-11-28b | The graph ground truth the lane score uses (`graph_gt_val.pkl` of the feature set b123) and its poses cannot be made from this repository: their scripts (`make_graph_gt.py`, `make_poses.py`) are in the tag `v1` only, and the records name that ground truth by file name, not by a signature. Bring them back behind the `GtGenerator` interface? |
