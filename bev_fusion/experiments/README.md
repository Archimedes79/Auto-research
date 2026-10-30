# experiments - one folder per experiment of bev_fusion

Each folder holds one growing file `experiment.md` (card -> detail plan -> summary, in the form of `../../autoresearch/templates/experiment.md`). Ids follow the stages of `../plan.md` section 11 (S1a = first experiment of stage S1).

| folder | question | status |
|---|---|---|
| `t6_soa_gap/` | why is the system not at the state of the art? GaussianCaR scored with OUR metric on our val frames (maps), AP-loss oracles on the stored detections, factor table with measured vs candidate causes, ranked decisive experiments | maps measured 2026-10-30 (ours 17-85 % of the SOA IoU, underfit, not a far-range problem); detection breakdown done, false-positive classification running |
| `t5_lane_probe/` | do the frozen image features hold the lane lines? image-plane probe, one identical head per feature source (DualBEV context / neck / ResNet stages, RGB control, prior) | v1 inconclusive (control failed, labels miss paint); v2 (2026-10-30, fresh frames, 2-30 m, F1): TIE - res4 +0.043 [+0.024, +0.061] over the lift input, below the 0.05 rule; prior already 0.44; picture: stride 4 sharp, stride 16 blurred |
| `t1_full_chain/` | first whole-chain training on nuScenes (camera + radar, dt -0.5 / 0 / +0.5 s) and two refinement rounds; report and result images | running (2026-10-26) |
| `s1a_lift_pd_vs_p/` | front camera, frozen DualBEV heads: lift PD (depth-weighted pull) against lift P (no depth) on nuScenes VAL, IoU by class and range; its synthetic unit test of the same pipeline; idle-GPU timing per block | plan (2026-10-26) |
