# experiments - one folder per experiment of bev_fusion

Each folder holds one growing file `experiment.md` (card -> detail plan -> summary, in the form of `../../autoresearch/templates/experiment.md`). Ids follow the stages of `../plan.md` section 11 (S1a = first experiment of stage S1).

| folder | question | status |
|---|---|---|
| `t5_lane_probe/` | do the frozen image features hold the lane lines? image-plane probe, one identical head per feature source (DualBEV context / neck / ResNet stages, RGB control, prior) | inconclusive (2026-10-30): the control failed; nuScenes divider labels miss much visible paint; v2 proposed |
| `t1_full_chain/` | first whole-chain training on nuScenes (camera + radar, dt -0.5 / 0 / +0.5 s) and two refinement rounds; report and result images | running (2026-10-26) |
| `s1a_lift_pd_vs_p/` | front camera, frozen DualBEV heads: lift PD (depth-weighted pull) against lift P (no depth) on nuScenes VAL, IoU by class and range; its synthetic unit test of the same pipeline; idle-GPU timing per block | plan (2026-10-26) |
