# E279: the soft close-node step at radius 1.25 m, trained end to end

Result record of experiment E279 (card `experiments/E279_soft-close-node-step-r125/experiment.md`, summary in its section 7; ledger rows `E279.w_s0`, `E279.w_s1`, `E279.wq_s0`, `E279.wq_s1`). It is E277 (radius 1.5 m, CUT by the guards) with the radius 1.25 m and nothing else changed. Baseline B = the E273 checkpoints (D58). The scoring cards and the checkpoints are not in git; this page names them and how to rebuild the numbers. All numbers VAL 873 frames, in-sample, no TEST.

## Verdict
**KEEP (the soft step replaces the hard read-out)** (`qgraph_run_eval.py verdict277 --exp-var e279`, by the scheduler 2026-11-08 06:06, the rule of the E277 card unchanged and written before the run): topo F1 gain over B above +0.02 and junction F1 gain above +0.03 on both seeds, every guard held, topo F1 within 0.01 of the hard step of E275 on both seeds. Thin margins on seed 0 (topo gain +0.0210 against 0.02, F1@1.0 -0.0097 against -0.010, lanes found -0.026 against -0.030): two seeds, in-sample, not a headline number.

## Numbers (seed 0 / seed 1; scoring cards `runs/e279/e279_w_s<k>.plain_t0.5/card.json`)
| | B | B + hard step 1.5 m (E275) | soft 1.5 m (E277, CUT) | soft 1.25 m (E279) | E279 - B |
|---|---|---|---|---|---|
| topo F1 | 0.597 / 0.595 | 0.626 / 0.625 | 0.621 / 0.624 | 0.618 / 0.620 | +0.021 / +0.025 |
| junction F1 | 0.130 / 0.120 | 0.271 / 0.249 | 0.263 / 0.268 | 0.223 / 0.218 | +0.094 / +0.098 |
| F1@1.0 | 0.832 / 0.821 | 0.828 / 0.816 | 0.818 / 0.820 | 0.823 / 0.822 | -0.0097 / +0.0006 |
| lanes found | 0.644 / 0.630 | 0.627 / 0.614 | 0.604 / 0.619 | 0.618 / 0.632 | -0.026 / +0.002 |
| vertices per 100 m (DP; budget 14.3) | 28.9 / 31.5 | 19.3 / 20.1 | 19.1 / 18.8 | 21.4 / 20.9 | |
| node recall / precision | 0.833 / 0.873, 0.833 / 0.868 | 0.793 / 0.915, 0.793 / 0.910 | 0.787 / 0.916, 0.790 / 0.911 | 0.805 / 0.907, 0.804 / 0.901 | |
| assignment p - g exact / error | -0.0072 / +0.0722, -0.0061 / +0.0706 | -0.0066 / +0.0731, -0.0058 / +0.0712 | -0.0064 / +0.0736, -0.0048 / +0.0715 | -0.0072 / +0.0734, -0.0059 / +0.0718 | |
With the 1.5 m hard step on top of the 1.25 m head (read-out close:1.5@0.5): topo F1 0.618 / 0.622, junction F1 0.253 / 0.249, lanes found 0.608 / 0.625, vertices per 100 m 20.0 / 19.5.

## How the numbers were made
| | |
|---|---|
| code | `run.py` snapshots, tags `lcr/E279.w_s0`, `lcr/E279.w_s1`; the step `qnodes.soft_suppress_logit` (commit 66cc4084, critic pass done; the gate of phase 3 reads the damped existence) |
| checkpoints (not in git) | `runs/e279_w_s0`, `runs/e279_w_s1` (phases 3-7 from `runs/e273_p_s<k>_p2/model.pt`; the setting `args['soft_nms'] = [1.25, 0.1, 3]`) |
| lane GT | lane GT l4, signature `0702a065c6` |
| training | `qgraph_train.py --name e279_w_s<k> --seed <k> --workers 3 --phases 3 4 5 6 7 --resume-from runs/e273_p_s<k>_p2/model.pt --gate-phases 3 --gate-node-prec 0.75 --gate-node-rec 0.5 --soft-nms 1.25 0.1 3` |
| scoring | `qgraph_run_eval.py score --run e279_w_s<k> --seed <k> --exp e279 --readouts plain@0.5 close:1.5@0.5` |
| verdict | `qgraph_run_eval.py verdict277 --var e279_w_s0.plain_t0.5 e279_w_s1.plain_t0.5 --ctl e273_p_s0 e273_p_s1 --hard e273_p_s0.close1.5_t0.5 e273_p_s1.close1.5_t0.5 --exp-var e279` |
| run notes | both trainings were stopped twice by a host memory failure (OSError ENOMEM in the DataLoader workers, kernel page allocation failure order 4) and resumed from their checkpoints; the ledger notes of the rows say how |
