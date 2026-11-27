# HANDOVER - bev_fusion (rewritten, not appended)

**State 2026-11-27.** The problem runs in the research system (restructure D154 / D155). Its code is the champion candidate's (`t21_ctl`,
the t20 network continued 1500 steps); the evaluation is warm (D70) on 1000 frames spread over the 82 VAL scenes, ground truth v13, score
`board.board7` (the mean of six map rows and the object mAP). Measured again through the new evaluation (a diagnostic, no card):
`t21_ctl_s0` board7 0.4631. The older 0.4549 is a cold read on ground truth v8 and not comparable. The problem's tests pass
(`research tests bev_fusion`; the trainkit equivalence needs a checkout that the git of WSL can read).

**Next.**
1. The owner's choice of the champion (`DECISIONS.md`, Open): `t21_ctl` or `t20_base_s0`; then a measurement experiment (kind measure,
   `import` of the run) and `research register`.
2. Against the literature this family is behind GaussianCaR on the map rows and behind CRN on objects (the t26 G0 tables, tag `v1`): the
   one long run t26 and the student encoder were the open lines.

**Know.** The code works in `<data root>/bev_fusion/` (configs name their files relative to it; `net/netpath.py` SITE). On the GPU a repeated
evaluation differs by up to 3e-3 (`anchor_tolerance`); the CPU is deterministic but takes about 85 minutes for 1000 frames, the GPU about 21.
Building the model loads the DualBEV weights in `models/pretrained/` every time. The split files name ground truth v8 (historical); the
site's cache is v13 (a107e0b4e8).
