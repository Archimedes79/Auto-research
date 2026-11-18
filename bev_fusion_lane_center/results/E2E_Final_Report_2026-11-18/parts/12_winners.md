# 12. The winning systems

Two systems are winners in the sense of this report: the best system with the earlier junction machinery, which is the end-to-end reference, and the token champion (D114). Both are small enough to be handed over as source folders without weights.

Table: The winning systems side by side (VAL 873 frames, seed 0 / seed 1).

| | E324 s, E2E read-out (reference) | token champion: the head with none weight 0.25 at its TRAIN-chosen decode (E373) |
|---|---|---|
| topo F1 | 0.6912 / 0.6948 | **0.6679 / 0.6762 at the final per-class decode** (E375, D115; junction F1 0.252 / 0.245); 0.6698 / 0.6769 at the single threshold (fresh heads, E373); selection heads 0.6731 / 0.6728 (E367); floors 0.6359 / 0.6349, plain head 0.6259 / 0.6249; runner-up (none weight 1.0): 0.6617 / 0.6767, 0.6618 / 0.6733 |
| with the learned wiring head (E358) | 0.6931 / 0.6964 | not applicable |
| F1@1.0, lanes found | 0.8777 / 0.8786, 0.7653 / 0.7648 | 0.846 / 0.847, 0.719 / 0.721 at the final decode (0.8456 / 0.8461, 0.7274 / 0.7267 at the single threshold; runner-up 0.836 - 0.842, 0.675 - 0.691) |
| parameters | 2.741 M | 2.32 M (the frozen lane head B' 2.196 M + the head 0.127 M) |
| time per frame (whole path, quiet window, 24 frames, batch 1, bf16) | 19.7 ms | 16.6 - 16.8 ms (identical architecture to the runner-up, which was timed) |
| training stages | three, plus the attachment | two (D120): stage 1 the head on the frozen cache, stage 2 the head and the node head of B' together (3000 steps) |
| hard steps left | keypoint peaks + top-K 32, wiring rule (or a learned argmax), close-node step, thresholds, candidate gates | the top-K 512 budget, the allowed-pair gate, the output thresholds (short link 0.3, long link 0.2, keep 0.6, none gate off) |
| code in git | `training/lanegraph/` (the package of 2026-11-16, no weights); `winners/e324s_lanegraph/` | `winners/e365_token_champion/` (the head, the loss, the decode, the loader, the cache, the trainer, the evaluation and their tests; no weights; it needs the repository; the champion differs from the runner-up by the option `--none-weight 0.25` and the decode values) |
| weights and run folder | `runs/e324_s_s0`, `runs/e324_s_s1` | **since D120 the stage-2 weights `runs/e372_s4_s0` / `runs/e372_s4_s1` (head + node head of B', `b_state`); PENDING at the build: the export of the winners folder is refreshed to the S4 code path (`jlive_train.py`) and verified**; stage-1 champion heads `runs/e373_cb_s0` (cache 0, head seed 2) and `runs/e373_cb_s1` (cache 1, head seed 3), selection heads `runs/e367_b25_s0 / s1`; runner-up `runs/e365_s1_s0 / s1`, `runs/e371_cf_s0 / s1`; frozen lane head B' `runs/e292_a_s0 / s1`; token cache `runs/e365_cache/be292_a_{0,1}_K512_9ad1d4ce30_v1` (derived); none in git |

**The architecture of the champion.** See Figure 2 in chapter 2: the predicted nodes of the frozen lane head as tokens, soft pruning through the key bias, three self-attention layers with relative-geometry bias, the per-source softmax with a none slot as the wiring, a seven-way node-type softmax; the champion is stage S1 with the lane head frozen; S2 (past) and S3 (map) were measured and are a TIE each (no block taken), S4 (joint fine-tune) is PENDING.

**Where the code is.** Small winner folders in git (`winners/`), one per winner, each with the source files, a README with the exact command and the numbers to reproduce, and a manifest (commit, file hashes, where the weights are); the weights stay outside git.

**The champion changed during the night.** The first champion (D114) was the head with none weight 1.0 at link 0.1 (topo F1 0.6617 / 0.6767 and 0.6618 / 0.6733). The arm with none weight 0.25 (E367) tied with it inside the band, and its confirmation on two fresh head seeds (E373) satisfied the rule written before the data (a quality tie on both caches), so it replaced the first recipe; the first recipe is the confirmed runner-up and both are in the same winner folder (one option, `--none-weight`, and the decode values differ).

**Scorecard of the champion (by hand, against anchors written in the file before the scores; scale 1 to 5).** Mirko asked for the current solution to be scored by simplicity, code efficiency, compute time and performance; the lane lead wrote the anchors and the scores in `experiments/E2E_champion_scorecard.md`, the numbers are read from the cards.

Table: Scorecard of the champion (a judgement, not a measured rule and not the verdict of a card).

| axis | 1 means / 3 means / 5 means | what the numbers say | score |
|---|---|---|---|
| performance | at or below plain B / above plain B by more than the band and within 0.03 topo F1 of the best E2E read-out / at or above the best read-out, no weak metric | topo F1 0.6679 / 0.6762, about +0.042 / +0.051 over plain B (0.626 / 0.625); 0.02 below the previous system's E2E read-out; junction F1 0.25 against the plain head's 0.27 | 3.5 |
| compute time | slower and larger than E324 s / equal / at least 30 % cheaper at equal quality | whole path 16.6 - 16.8 ms against 19.7 ms (-15 %), 2.323 M against 2.741 M parameters (-15 %) | 4 |
| simplicity | many learned stages and hard steps / a few of each / one network, no hard step | one head, two softmaxes, none of the five rule steps of E324 s; three named hard steps remain | 3.5 |
| code efficiency | large, untested, many scratch scripts / the model is small, the tooling around it large / small model, small tooling, all promoted and tested | model, data and trainer about 800 lines, the evaluation tool 1126 lines; tested and promoted, the tooling is large | 3 |

The scores are a judgement by hand against anchors, not a measured rule. The mean is **3.5 of 5** (unweighted; the weights are the owner's call).

**Re-scored for the final report (a new line; the scorecard above stays as the history of the S1 champion, by D120).** The champion is now the two-stage S4 recipe; the lead asked for the same four axes to be scored again by hand against the same anchors. Performance and compute are scored on the cards; simplicity and code efficiency are scored after the landing of the clean-up (chapter 8 holds the counts) and are PENDING until then.

Table: The same four axes, scored again for the champion since D120 (a judgement by hand against the anchors of `experiments/E2E_champion_scorecard.md`).

| axis | what the numbers say now | score (before / now) |
|---|---|---|
| performance | topo F1 0.6682 / 0.6813 (E372, read live at the D115 cell), +0.042 / +0.056 over plain B (0.626 / 0.625), 0.023 / 0.014 below the previous system's E2E read-out (0.6912 / 0.6948), inside the 0.03 of anchor 3; junction F1 0.2538 / 0.2481 still below the plain head's 0.27, so anchor 5 (no weak metric) is not met | 3.5 / 3.5 (the gain over the staged champion, +0.0002 / +0.0051, is inside the noise of one seed and does not move a half point) |
| compute time | the inference network is the same: 16.6 - 16.8 ms, 2.323 M parameters (-15 % against E324 s on both); the training of the head takes a stage 2 of 3000 steps at about 0.29 s per step against 0.24 s (B' runs in the loop) | 4 / 4 |
| simplicity | the staged gap "B' frozen" is closed for the node head (the loss reaches it); the recipe has two stages; the hard steps stay named (top-K 512, the allowed-pair gate, four output thresholds); the clean-up removed the S2 / S3 blocks and the recursive-state families (chapter 8) | 3.5 / PENDING after the landing |
| code efficiency | PENDING: the counts of chapter 8 after the landing (files and lines of the groups, the suite of chapter 9) | 3 / PENDING after the landing |

**Reading of the scorecard.** The champion buys **simplicity and 15 % compute** for **about 0.02 topo F1** against the previous winner's E2E read-out, and it is **+0.044 / +0.052 over plain B** (single-threshold decode). It is not the best number of the project (E324 s snapped 0.698 / 0.701 needs rule steps); it is the best number of the solution set that carries no wiring rule. **What to improve, in order:** (1) the junction F1 (0.25 against 0.27 for plain B after E375) and precision; E376 (a max-positive wiring loss) was a NO and S2 (the past) a TIE, S3 (the map) a TIE, and the placement of the none push (E377) was TARGETING-ONLY, so what is left is the junction F1 through what the head sees (S4); (2) S4, so that "staged, B' frozen" disappears; (3) a real SD map instead of the simulation, if S3 is to be tested for more than an upper bound (S3 is done: a TIE); (4) code: promote the scratch scripts and move the diagnostics of the 1126-line evaluation tool out of the delivered path. Not scored: TEST (only with the owner's recorded OK) and the stage S4 (not run); S2 and S3 are measured (a TIE each).
