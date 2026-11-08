# E277: the close-node suppression as a soft step inside the lane head, trained end to end

Result record of experiment E277 (card `experiments/E277_soft-close-node-step/experiment.md`, summary in its section 7; ledger rows `E277.u_s0`, `E277.u_s1`, `E277.uq_s0`, `E277.uq_s1`; the diagnostic E278 in section 7a, rows `E278.qp_s0`, `E278.qp_s1`). Baseline B = the E273 checkpoints (D58). The scoring cards, the checkpoints and the pictures are not in git; this page names them and how to rebuild the numbers. All numbers VAL 873 frames, in-sample, no TEST.

## Verdict
**CUT by the written guards** (`qgraph_run_eval.py verdict277`, by hand, the rule of the card written before the run; run 2026-11-08 03:40). Seed 0 breaks two guards (F1@1.0 -0.0147 against -0.010, lanes found -0.0394 against -0.030), seed 1 holds all. Both headline gains hold on both seeds (topo F1 +0.024 / +0.029, junction F1 +0.133 / +0.147) and the soft step replaces the hard step of E275 (topo F1 trail 0.005 / 0.001). All five numeric predictions of the card held. The hard step stays the named read-out. Follow-up E279 (radius 1.25 m, same rule).

## Numbers (seed 0 / seed 1; scoring cards `runs/e277/e277_u_s<k>.plain_t0.5/card.json`)
| | B | B + hard step (E275) | head with the soft step | soft - B |
|---|---|---|---|---|
| topo F1 | 0.597 / 0.595 | 0.626 / 0.625 | 0.621 / 0.624 | +0.024 / +0.029 |
| junction F1 | 0.130 / 0.120 | 0.271 / 0.249 | 0.263 / 0.268 | +0.133 / +0.147 |
| F1@1.0 | 0.832 / 0.821 | 0.828 / 0.816 | 0.818 / 0.820 | -0.015 / -0.001 |
| lanes found | 0.644 / 0.630 | 0.627 / 0.614 | 0.604 / 0.619 | -0.039 / -0.011 |
| vertices per 100 m (DP; budget 14.3) | 28.9 / 31.5 | 19.3 / 20.1 | 19.1 / 18.8 | |
| node recall / precision | 0.833 / 0.873, 0.833 / 0.868 | 0.793 / 0.915, 0.793 / 0.910 | 0.787 / 0.916, 0.790 / 0.911 | |
| long-link precision | 0.167 / 0.191 | 0.172 / 0.194 | 0.172 / 0.186 | |
| assignment p - g exact / error | -0.0072 / +0.0722, -0.0061 / +0.0706 | -0.0066 / +0.0731, -0.0058 / +0.0712 | -0.0064 / +0.0736, -0.0048 / +0.0715 | |
The hard step on top of the soft step removes no further node (the read-outs plain@0.5 and close:1.5@0.5 of the soft head give identical cards).

## How the numbers were made
| | |
|---|---|
| code | `run.py` snapshots, tags `lcr/E277.u_s0`, `lcr/E277.u_s1`; the step `qnodes.soft_suppress_logit` and `QGraph(soft_nms=...)` in commit 66cc4084 (critic pass done; the two trainings ran the code of before the critic fixes: raw-head phase-3 gate, a 1e-6 clamp in the step) |
| checkpoints (not in git) | `runs/e277_u_s0`, `runs/e277_u_s1` (phases 3-7 from `runs/e273_p_s<k>_p2/model.pt`; the setting is in `args['soft_nms']`) |
| lane GT | lane GT l4, signature `0702a065c6` (geometry equal to l3 `3576fa1f7e`) |
| training | `qgraph_train.py --name e277_u_s<k> --seed <k> --workers 3 --phases 3 4 5 6 7 --resume-from runs/e273_p_s<k>_p2/model.pt --gate-phases 3 --gate-node-prec 0.75 --gate-node-rec 0.5 --soft-nms 1.5 0.1 3` |
| scoring | `qgraph_run_eval.py score --run e277_u_s<k> --seed <k> --exp e277 --readouts plain@0.5 close:1.5@0.5` |
| verdict | `qgraph_run_eval.py verdict277 --var e277_u_s0.plain_t0.5 e277_u_s1.plain_t0.5 --ctl e273_p_s0 e273_p_s1 --hard e273_p_s0.close1.5_t0.5 e273_p_s1.close1.5_t0.5` (`temp/e276/verdicts.log`) |
| pictures | `temp/e276/compare_views.py <dir> <frames> e273_p_s0 e273_p_s0+close e277_u_s0 e276_t_s0` (CPU), `results/E277/figures/` (not in git) |
