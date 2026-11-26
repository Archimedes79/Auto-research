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
