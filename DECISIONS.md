# Decisions that concern more than one problem

The owner's decisions about the research system as a whole (its gates, git and push, data, the machines). A problem's own decisions are in its
`problems/<name>/DECISIONS.md`; the ids `D<n>` are one count across all of them. A question waits in the Open table until the owner answers;
an agent records the answer as a decided row and never acts before. The rows up to D153 are in the tag `v1`
(`bev_fusion_lane_center/experiment_execution/DECISIONS.md`, which then held the decisions of both problems and of the system).

## Decided
| date | decision |
|---|---|
| 2026-11-26 | D154: the owner's orders of 2026-11-26 (paraphrased): carry out the restructure plan `autoresearch/INTEGRATION_2026-11-26.md` steps S1 - S5 in a branch; build the new system cleaner and more extensible, test the tools, clean up the code in the branch, review architecture and code, keep it clean and free of bugs; main stays the old system and what the branch removes stays restorable, tagged `v1`; on old data "keep only what is needed": the champion chains and the comparison networks keep their weights and predictions, other checkpoints, caches and run outputs may be deleted after a catalog lists them (this changes hard rule 5 for the restructure). Applied: branch `v2` in its own worktree, protected files and the gate hook moved and rewritten as the plan says, the TEST logs carried over unchanged. |
| 2026-11-27 | D155: the owner's orders of 2026-11-27 (paraphrased): finish the plan unattended; test the system with the lane-centre problem; review the system and the code, keep it clean and structured, update the documentation; remove dead code, old experiment data, old markdown files, old scratch and temp files, so that git and the folder are clean; complete the tools; push the branch and merge it into main when done. Applied: what is deleted is listed with its size before, in `autoresearch/INTEGRATION_2026-11-26.md` section 13; raw data, the weights of the champions and their chains, and the data they read stay (moved to the data root on D:). |

## Open
| id | question |
|---|---|
