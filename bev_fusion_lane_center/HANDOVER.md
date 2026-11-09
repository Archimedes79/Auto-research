# HANDOVER - 2026-11-09 09:00

New chat: start in the repository folder, say "Read HANDOVER.md and continue". Rewrite at every milestone (<= 3 KB). Mirko reads little: plain terms (B = baseline only). **D57 + D61: this (Spurmitte lane) chat has priority.** GPU only in his windows (nights 22-07, Mon-Fri 08:30-17:30). **D60: when Mirko says he is away, work alone.** No TEST, push, mail, shutdown. Do not edit `autogt/` (GT chat).

## Results (cards E275 - E283). System report: https://claude.ai/artifact/9ShDVSELV2gi6xLLcGmpVF (results/system_report_2026-11-08)
- **E275 KEEP** hard close-node step; **E279** soft step 1.25 m KEEP (**open for Mirko, no D-id:** adopt it instead of the hard read-out, changes B, D58). E276 t TIE / s failed; E277 CUT; E280 / E281 / E282 closed by their CPU gates; D64 long-link threshold stays 0.5.
- **E283 (analysis card, CPU only, 2026-11-09; `experiments/E283_junction-error-analysis/experiment.md`):** after Mirko's marks on the E279 centre map. The junction blur = the pixel map hedges between unknown entry -> exit PAIRS (oracle from GT ends: true pairs 0.88, all pairs 0.53, network 0.71); geometry of clean ends predicts the pairs (AUC 0.97 - 0.995, only ~35 junction groups in the data). The delivered ends are NOT clean (flagged ends within 2 m of a GT end 49 %, within 4 m 73 %). Largest graph loss = a local lateral offset of 1 - 2.5 m (11.5 % of the GT length) that is in the head's own map (shared by both seeds; per-frame rigid shift removes 14 %; ridge snapping +0.004). The pair-rule label undercounts the long links (10 % vs 46 - 52 % geometrically right). **Closed by written gates:** geometry on long links with the pair label (p14), re-scored read-out (p16), ridge snapping, long-link NMS, tiny pieces. p15 (strict label) passed its gate but adds only +0.01 - 0.03 AUC over the strong current features.
- Options, none decided: end keypoint head with along-lane tolerance, pair decision with geometry in the link head, CPU scorer oracle (GT ends + pair classifier + one curve per pair under the E273 scorer) FIRST; features / unfreezing belong to the encoder chats.

## State of the machine
Lane chain done. Lane GPU job needs the other chats told first; GPU busy with BEV system / Image Encoder jobs at night (t21 / B2 / t23). GT chat merged l5 into live autogt/ (store `0702a065c6` = l4 still the default; l5 `db288acca5`); wait for its 'live checks done' before a new lane job. WSL page-cache ENOMEM: `temp/e276/reclaim.py`. Scratch of E283: `temp/e284/` (tag `lcr/e284-scratch-1`).

## If continuing
Mirko reads the E283 summary first (options above), then picks. Nothing pushed.
