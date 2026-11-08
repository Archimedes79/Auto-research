# E276: lane GT l4 (relevance tiers, visibility), a tier-weighted loss and a sparse node target for the lane head

Result record of experiment E276 (card `experiments/E276_gt-tiers-sparse-nodes-tier-loss/experiment.md`, summary in its sections 7a / 7b; ledger rows `E276.c_s0`, `E276.c_s1`, `E276.t_s0`, `E276.t_s1`, `E276.tq_s0`, `E276.tq_s1`, `E276.s_s0`, smokes `E276.mt_s0`, `E276.ms_s0`). Baseline B = the E273 checkpoints (D58) scored on lane GT l4. The scoring cards, the dumps and the pictures are not in git; this page names them and how to rebuild the numbers. All numbers VAL 873 frames, in-sample, no TEST.

## Verdicts
- **Arm t (tier-weighted loss on the dense node target) against B: TIE, not adopted** (`verdict276` kind t, by hand, the written rule of the card; both seeds on the same side; guards held). The tier weights do not lift the own lane and cost the far lanes.
- **Arm s (tier loss + sparse node target 0.3 m / 7.5 m): failed the gate of phase 3 (seed 0), CUT by the card's kill criterion.** Node recall within 2 m 0.143 (floor 0.5). No s seed 1, no scoring.

## Numbers (seed 0 / seed 1; scoring cards `runs/e276/<run>/card.json`)
| | B | arm t | gain |
|---|---|---|---|
| t12_F1_1 (F1 at 1 m, own lane + neighbours, visible parts) | 0.888 / 0.875 | 0.886 / 0.885 | -0.002 / +0.010 (band 0.02) |
| own-lane (tier 1) recall | 0.851 / 0.811 | 0.838 / 0.842 | -0.013 / +0.031 |
| far-lane (tier 3) recall | 0.796 / 0.769 | 0.698 / 0.670 | -0.098 / -0.099 |
| topo F1 | 0.597 / 0.595 | 0.580 / 0.573 | -0.017 / -0.022 |
| lanes found | 0.644 / 0.630 | 0.557 / 0.529 | -0.087 / -0.101 |
| node precision / recall | 0.873 / 0.833, 0.868 / 0.833 | 0.892 / 0.804, 0.888 / 0.790 | |
| false nodes in the own-lane zone | 10.2 % / 10.1 % | 10.1 % / 10.8 % | none |
| delivered nodes per 100 m | 31.5 / 31.7 | 29.7 / 29.4 | -5.6 % / -7.4 % |
| assignment p - g, exact map / error 1-2 m | -0.0072 / +0.0722, -0.0061 / +0.0706 | -0.0044 / +0.0778, -0.0058 / +0.0731 | guards held |
Arm s diagnostic (CPU, 60 frames, the phase-3 checkpoint against the sparse GT; cards `runs/e276diag/`): delivered nodes per 100 m 3.6 at e >= 0.5 (recall 0.146), 50.0 at e >= 0.3 (recall 0.910, precision 0.310), 71.8 at e >= 0.2; target 14.4. The head knows where the nodes are but not which of about three cells along a straight lane carries the node (a hypothesis; section 7b of the card).

## CPU analyses of Mirko's eight video notes (card section 7a; full dumps of B `temp/e276/predfull_e273_p_s<k>.pkl`)
P1 sparse target keeps the oracle geometry (56 % fewer nodes, vertices per 100 m 11.4 for a perfect network under any node target); P2 labels (tier 1 16 %, tier 2 7 % of the lane length); P3 B per tier; P4 map-extension rule recall 0.92 - 0.95; P5 / P5b flicker (nodes: 5.3 % near intersections against 1.7 %; links: long links persist 64 - 65 % against 98 % in the GT); P6 endpoints; P7 intersection polygon covers 77 % of the in-intersection connectors; P8 / P8b assignment (97.6 % per node, 88 % of map lanes claimed or shared, 4.6 % misassigned); P9 link calibration (over-confident, long links precision 0.17 - 0.19); P10 link thresholds on the graph numbers (a trade, no gain; without the long links topo F1 -0.12).

## How the numbers were made
| | |
|---|---|
| code | `run.py` snapshots, tags `lcr/E276.<id>`; scratch (scheduler `temp/e276/launch_chain.sh`, start scripts, analyses `exp_*.py`, report builder) in the tags `lcr/e276-scratch*` |
| lane GT | lane GT l4, signature `0702a065c6` (autogt tag `autogt/lane-l4`, release v11; geometry equal to l3 `3576fa1f7e`); front GT of the cached features `c643a7539d` |
| checkpoints | `runs/e276_t_s0`, `runs/e276_t_s1` (phases 3-7 from `runs/e273_p_s<k>_p2/model.pt`), `runs/e276_s_s0_p3` (not in git) |
| training | `qgraph_train.py --name e276_t_s<k> --seed <k> --workers 3 --phases 3 4 5 6 7 --resume-from runs/e273_p_s<k>_p2/model.pt --gate-phases 3 --gate-node-prec 0.75 --gate-node-rec 0.5 --tier-w 3 1.5 1 --vis-w 0 --neg-w 2 1.5 1 1 --link-tier-w 3 1.5 1` (arm s adds `--sparse 0.3 7.5`) |
| scoring | `qgraph_run_eval.py score --run e273_p_s<k> --seed <k> --exp e276 --dump-full` (B), `--run e276_t_s<k>` (arm t) |
| verdict | `qgraph_run_eval.py verdict276 --kind t --var e276_t_s0 e276_t_s1 --ctl e273_p_s0 e273_p_s1` (written by the scheduler, `temp/e276/verdicts.log`) |
