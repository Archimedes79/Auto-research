# E274: node recall weights and long-link matching on the E273 pyramid head

Result record of experiment E274 (card `experiments/E274_node-recall-and-long-link-matching/experiment.md`, summary in its section 7; ledger rows `E274.r_s0`, `E274.r_s1`, `E274.p_s0`, `E274.p_s1`). Baseline B = the E273 checkpoints (D58). The scoring cards, the checkpoints and the scratch are not in git; this page names them and how to rebuild the numbers.

## Verdict
**Arm r against B: CUT. Arm p against r: CUT. Arm p against B: CUT** (by hand, the written rule of the card, `qgraph_run_eval.py verdict274`, both seeds on the same side). B stays the baseline. Not adopted.

## Numbers (VAL 873 frames, from the scoring cards `runs/e274/<run>/card.json`, `runs/e273/<run>/card.json`)
| | B s0 | B s1 | r s0 | r s1 | p s0 | p s1 |
|---|---|---|---|---|---|---|
| topo F1 | 0.597 | 0.595 | 0.558 | 0.557 | 0.582 | 0.577 |
| F1@1.0 | 0.832 | 0.821 | 0.865 | 0.855 | 0.863 | 0.852 |
| junction F1 | 0.130 | 0.120 | 0.121 | 0.121 | 0.124 | 0.122 |
| lanes found | 0.644 | 0.630 | 0.777 | 0.756 | 0.752 | 0.736 |
| Q | 0.636 | 0.630 | 0.647 | 0.642 | 0.648 | 0.641 |
| duplicates (dup_frac) | 0.232 | 0.192 | 0.363 | 0.340 | 0.321 | 0.281 |
| topo precision / recall | 0.586 / 0.609 | 0.600 / 0.590 | 0.467 / 0.693 | 0.477 / 0.668 | 0.510 / 0.680 | 0.513 / 0.660 |
| vertices per 100 m (DP) | 28.9 | 31.5 | 27.5 | 28.2 | 28.6 | 29.9 |
| node recall / precision | 0.833 / 0.873 | 0.833 / 0.868 | 0.889 / 0.811 | 0.876 / 0.814 | 0.888 / 0.811 | 0.876 / 0.814 |
| long-link precision / recall | 0.167 / 0.503 | 0.191 / 0.470 | 0.179 / 0.518 | 0.190 / 0.509 | 0.216 / 0.486 | 0.221 / 0.469 |
| assignment p - g, map error 1-2 m (guard >= +0.03) | +0.0722 | +0.0706 | +0.0625 | +0.0585 | +0.0618 | +0.0590 |
| assignment p - g, exact map (guard >= -0.01) | -0.0072 | -0.0061 | -0.0173 | -0.0160 | -0.0164 | -0.0156 |

Reference d (recall_sp0_dp): topo F1 0.741, F1@1.0 0.878, junction F1 0.273, duplicates 0.077. Verdict gains: r - B topo F1 -0.039 / -0.038, F1@1.0 +0.033 / +0.033; p - r junction F1 +0.003 / +0.0002, topo F1 +0.025 / +0.020; p - B topo F1 -0.015 / -0.018. The read-out diagnostic (close-node suppression, in-sample) is in the card, section 7, and continues in `experiments/E275_node-readout-suppression/`.

## How the numbers were made
| | |
|---|---|
| code | tags `lcr/E274.r_s0`, `lcr/E274.r_s1`, `lcr/E274.p_s0`, `lcr/E274.p_s1` (run.py snapshots); scratch scripts (diagnostics, read-out sweeps, launchers, ledger note script) in tag `lcr/e274-scratch-2` (older: `lcr/e274-scratch`, without the sweep script); logs and json outputs of the scratch are not kept (git rule): the sweep numbers are in the card, section 7 |
| checkpoints (sha256, first 12 hex; not in git) | `runs/e274_r_s0/model.pt` b3f60f00c714, `runs/e274_r_s1/model.pt` 4e557acc3c26, `runs/e274_p_s0/model.pt` a3fc7f98d60b, `runs/e274_p_s1/model.pt` 33d1d9e69b07 (p starts from r's phase-3 models `runs/e274_r_s<k>_p3/model.pt`) |
| lane GT | `autogt/lane-l3`, signature `3576fa1f7e`; front GT of the cached features `c643a7539d` |
| evaluation | VAL 873 frames (the list of E273), evaluator `evaluation/qgraph_run_eval.py` (last commit before the scoring `ff8ba947`) |
| scoring | `wsl bash tools/env/_py14.sh qgraph_run_eval.py score --run e274_r_s0 --seed 0 --exp e274` (likewise r_s1, p_s0, p_s1) |
| verdicts | `qgraph_run_eval.py verdict274 --var e274_r_s0 e274_r_s1 --ctl e273_p_s0 e273_p_s1`; p against r: `--var e274_p_s0 e274_p_s1 --ctl e274_r_s0 e274_r_s1 --exp-ctl e274 --metric junction_F1 --band 0.03`; p against B as the first with p |
| read-out sweeps | `temp/e274/sweep_gpu.py <run> [--skip-plain]`, outputs `temp/e274/sweep_<run>.json` (scratch, not kept; rebuild by running the script); the 3 x 3 cell rule in it is invalid and was discarded |
| gate caution | the live GATE line at the end of a phase and the phase-7 record are different models; compare like with like (card, section 7) |
