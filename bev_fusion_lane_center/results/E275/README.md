# E275: close-node suppression as a named read-out step on the E273 baseline

Result record of experiment E275 (card `experiments/E275_node-readout-suppression/experiment.md`, summary in its section 7; ledger rows `E275.n_s0`, `E275.n_s1`, `E275.rn_s0`, `E275.rn_s1`). Baseline B = the E273 checkpoints (D58). No retraining: the step is applied to the delivered nodes of existing checkpoints. The scoring cards, the dumps and the pictures are not in git; this page names them and how to rebuild the numbers. All numbers are VAL 873 frames and in-sample (the radius 1.5 m and the threshold were seen on VAL in the E274 sweep); no TEST use.

## Verdict
**KEEP the step as the named read-out of the lane head** (by hand, `qgraph_run_eval.py verdict275`, the rule of the card section 0 written before the run; both seeds on the same side). It is a D55 stop-gap (hard selection on the output), not part of the network. The system pass (topo F1 within 0.03 of d, 0.7105) is not reached.

## Numbers (from the scoring cards `runs/e275/<run>.<readout>/card.json`)
| | B s0 | B + step s0 | B s1 | B + step s1 | r s0 | r + step s0 | r s1 | r + step s1 |
|---|---|---|---|---|---|---|---|---|
| topo F1 | 0.597 | 0.626 | 0.595 | 0.625 | 0.558 | 0.614 | 0.557 | 0.613 |
| junction F1 | 0.130 | 0.271 | 0.120 | 0.249 | 0.121 | 0.235 | 0.121 | 0.242 |
| F1@1.0 | 0.832 | 0.828 | 0.821 | 0.816 | 0.865 | 0.861 | 0.855 | 0.851 |
| lanes found | 0.644 | 0.627 | 0.630 | 0.614 | 0.777 | 0.748 | 0.756 | 0.728 |
| Q | 0.636 | 0.667 | 0.630 | 0.656 | 0.647 | 0.668 | 0.642 | 0.664 |
| duplicates (dup_frac) | 0.232 | 0.234 | 0.192 | 0.201 | 0.363 | 0.351 | 0.340 | 0.329 |
| node recall / precision | 0.833 / 0.873 | 0.793 / 0.915 | 0.833 / 0.868 | 0.793 / 0.910 | 0.889 / 0.811 | 0.841 / 0.866 | 0.876 / 0.814 | 0.831 / 0.870 |
| vertices per 100 m (DP; budget 14.3) | 28.9 | 19.3 | 31.5 | 20.1 | 27.5 | 20.6 | 28.2 | 20.7 |
| assignment p - g, exact map (guard >= -0.01) | -0.0072 | -0.0066 | -0.0061 | -0.0058 | -0.0173 | -0.0157 | -0.0160 | -0.0148 |
| assignment p - g, error 1-2 m (guard >= +0.03) | +0.0722 | +0.0731 | +0.0706 | +0.0712 | +0.0625 | +0.0645 | +0.0585 | +0.0601 |
| nodes dropped by the step | - | 9.1 % | - | 9.2 % | - | 11.4 % | - | 11.2 % |
| ms per frame of the step | - | 1.26 | - | 1.75 | - | 1.63 | - | 1.48 |

Verdict gains (step minus plain, B): topo F1 +0.029 / +0.030, junction F1 +0.142 / +0.129, F1@1.0 -0.004 / -0.005, lanes found -0.017 / -0.017. Arm r + step is reported, not judged (exact-map guard broken, -0.0157 / -0.0148). Radius curve, threshold 0.4 read-outs, the analysis of the dropped nodes (65 % of them counted true by the scorer; 93-95 % of the true nodes lost have another GT node within 1.5 m) and the five predictions (two held, three missed, rn missed prediction 1 as well) are in the card, section 7.

## How the numbers were made
| | |
|---|---|
| code | `run.py` snapshots (ledger column `commit`): E275.n_s0 `507c0543cf` (the tag `lcr/E275.n_s0` names attempt 1, `b783d03499`: the first attempt of 2026-11-05 failed on a relative `--repro` path), E275.n_s1 `2f9e41f5f9`, E275.rn_s0 `0814225d5f`, E275.rn_s1 `11066ca83c` (tags `lcr/E275.<id>`); scratch (launcher, analysis, summary scripts, ledger note scripts) in tag `lcr/e275-scratch` |
| checkpoints | the E273 and E274 checkpoints (hashes in `results/E273/README.md` and `results/E274/README.md`) |
| lane GT | lane GT l4, signature `0702a065c6` (autogt tag `autogt/lane-l4`; geometry equal to l3 `3576fa1f7e`); front GT of the cached features `c643a7539d` |
| evaluation | VAL 873 frames, evaluator `evaluation/qgraph_run_eval.py`, read-out `evaluation/qgraph_readout.py` (tests `evaluation/tests/test_qgraph_readout.py`, `test_verdict275.py`) |
| scoring | `wsl bash tools/env/_py14.sh qgraph_run_eval.py score --run e273_p_s0 --seed 0 --exp e275 --readouts plain@0.5 close:1.5@0.5 close:1.5@0.4 plain@0.4 --repro temp/e274/sweep_e273_p_s0.json --curve 1.0 1.25 1.75 2.0 --dump` (seed 1 likewise; arm r: `--run e274_r_s0 --readouts plain@0.5 close:1.5@0.5`); the reference files for `--repro` are tracked as `results/E274/reference/sweep_*.json` |
| verdict | `qgraph_run_eval.py verdict275 --var e273_p_s0.close1.5_t0.5 e273_p_s1.close1.5_t0.5 --ctl e273_p_s0 e273_p_s1` |
| analysis of the dropped nodes | `temp/e275/analyse.py <run> ...` on the dumps `temp/e275/pred_<run>.pkl` (scratch, tag `lcr/e275-scratch`); pictures `results/E275/figures/<run>_dropped_<k>.png` (not in git) |
