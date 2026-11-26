# Decisions of the BEV Fusion problem

The owner's decisions about this problem. A question waits in the Open table until the owner answers; an agent records the answer as a decided
row and never acts before. Ids `D<n>` are counted together with the root `DECISIONS.md`. Before 2026-11-27 this problem's decisions were rows of
the lane problem's record (tag `v1`, `bev_fusion_lane_center/experiment_execution/DECISIONS.md`); the technical ones live in `DESIGN.md`, the
owner's requirements in `PROBLEM.md`.

## Decided
| date | decision |
|---|---|
| 2026-11-10 | D70: a network with a memory is evaluated warm: whole scenes in time order from the network's own start. |
| 2026-11-26 | D154: the restructure into the research system (see the root `DECISIONS.md`); this folder holds the champion's code, moved from `bev_fusion/` with its history; the champion is measured again with the new evaluation. |

## Open
| id | question |
|---|---|
| - | Which network is the champion: `t21_ctl` (best by value, the default here) or `t20_base_s0` (the reference baseline of D48)? Both lie inside the band 0.02. |
