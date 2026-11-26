# HANDOVER - bev_fusion (rewritten, not appended)

**State 2026-11-27.** The problem runs in the research system (restructure D154 / D155). Its code is the champion's (`t21_ctl`, the t20 network
continued 1500 steps); the evaluation is warm (D70) on 1000 frames spread over the 82 VAL scenes, the score `board.board7` (the mean of six map
rows and the object mAP). The champion is measured again with this evaluation (the measurement experiment of the problem); the older board7
0.4549 is a cold read on ground truth v8 and not comparable.

**Next.**
1. The owner's choice of the champion (`DECISIONS.md`, Open): `t21_ctl` or `t20_base_s0`; then `research register`.
2. Against the literature this family is behind GaussianCaR on the map rows and behind CRN on objects (the t26 G0 tables, tag `v1`): the
   one long run t26 and the student encoder were the open lines.

**Know.** The code works in `<data root>/bev_fusion/` (configs name their files relative to it). On the GPU a repeated evaluation differs by up
to 3e-3 (`anchor_tolerance`); the CPU is deterministic but takes about 85 minutes for 1000 frames. Building the model loads the DualBEV
weights in `models/pretrained/` every time.
