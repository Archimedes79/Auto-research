# HANDOVER - bev_fusion (rewritten, not appended)

**State 2026-11-27.** The problem runs in the research system (restructure D154 / D155). Its code is the champion candidate's (`t21_ctl`,
the t20 network continued 1500 steps); the evaluation is warm (D70) on 1000 frames spread over the 82 VAL scenes, ground truth v13, score
`board.board7` (the mean of six map rows and the object mAP). Measured through the research system (E27, E28: weights imported, kind
measure): `t21_ctl` board7 0.4631 / 0.4635 (seed 0 / 1), `t20_base_s0` 0.4569 - inside the band 0.02 of each other. The older
0.4549 / 0.4465 are cold reads on ground truth v8 and not comparable. The problem's tests pass
(`research tests bev_fusion`).

**Next.**
1. The owner's choice of the champion (`DECISIONS.md`, Open): `t21_ctl` or `t20_base_s0`; then `research register bev_fusion E27 t21
   --decision D<n> --role champion` (or E28 t20); the other one as a comparison network.
2. Against the literature this family is behind GaussianCaR on the map rows and behind CRN on objects (the t26 G0 tables, tag `v1`): the
   one long run t26 and the student encoder were the open lines.

**Know.** The code works in `<data root>/bev_fusion/` (configs name their files relative to it; `net/netpath.py` SITE). On the GPU a repeated
evaluation differs by up to 3e-3 (`anchor_tolerance`); the CPU is deterministic but takes about 85 minutes for 1000 frames, the GPU about 21.
Building the model loads the DualBEV weights in `models/pretrained/` every time. The split files name ground truth v8 (historical); the
site's cache is v13 (a107e0b4e8). A TEST evaluation is not built yet (`evaluation/bev_sets.py` offers no test set), so
`research test bev_fusion` cannot finish.
