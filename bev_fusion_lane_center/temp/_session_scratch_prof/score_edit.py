p = "bev_fusion_lane_center/results/E2E_Final_Report_2026-11-18/parts/12_winners.md"
s = open(p, encoding="utf-8").read()
key = "**Reading of the scorecard.**"
assert s.count(key) == 1
new = """**Re-scored for the final report (a new line; the scorecard above stays as the history of the S1 champion, by D120).** The champion is now the two-stage S4 recipe; the lead asked for the same four axes to be scored again by hand against the same anchors. Performance and compute are scored on the cards; simplicity and code efficiency are scored after the landing of the clean-up (chapter 8 holds the counts) and are PENDING until then.

Table: The same four axes, scored again for the champion since D120 (a judgement by hand against the anchors of `experiments/E2E_champion_scorecard.md`).

| axis | what the numbers say now | score (before / now) |
|---|---|---|
| performance | topo F1 0.6682 / 0.6813 (E372, read live at the D115 cell), +0.042 / +0.056 over plain B (0.626 / 0.625), 0.023 / 0.014 below the previous system's E2E read-out (0.6912 / 0.6948), inside the 0.03 of anchor 3; junction F1 0.2538 / 0.2481 still below the plain head's 0.27, so anchor 5 (no weak metric) is not met | 3.5 / 3.5 (the gain over the staged champion, +0.0002 / +0.0051, is inside the noise of one seed and does not move a half point) |
| compute time | the inference network is the same: 16.6 - 16.8 ms, 2.323 M parameters (-15 % against E324 s on both); the training of the head takes a stage 2 of 3000 steps at about 0.29 s per step against 0.24 s (B' runs in the loop) | 4 / 4 |
| simplicity | the staged gap "B' frozen" is closed for the node head (the loss reaches it); the recipe has two stages; the hard steps stay named (top-K 512, the allowed-pair gate, four output thresholds); the clean-up removed the S2 / S3 blocks and the recursive-state families (chapter 8) | 3.5 / PENDING after the landing |
| code efficiency | PENDING: the counts of chapter 8 after the landing (files and lines of the groups, the suite of chapter 9) | 3 / PENDING after the landing |

"""
open(p, "w", encoding="utf-8", newline="\n").write(s.replace(key, new + key))
