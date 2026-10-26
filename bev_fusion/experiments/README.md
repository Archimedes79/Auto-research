# experiments - one folder per experiment of bev_fusion

Each folder holds one growing file `experiment.md` (card -> detail plan -> summary, in the form of `../../autoresearch/templates/experiment.md`). Ids follow the stages of `../plan.md` section 11 (S1a = first experiment of stage S1).

| folder | question | status |
|---|---|---|
| `s1a_lift_pd_vs_p/` | front camera, frozen DualBEV heads: lift PD (depth-weighted pull) against lift P (no depth) on nuScenes VAL, IoU by class and range; its synthetic unit test of the same pipeline; idle-GPU timing per block | plan (2026-10-26) |
