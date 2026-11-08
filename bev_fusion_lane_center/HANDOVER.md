# HANDOVER - 2026-11-08 03:55

New chat: start in the repository folder, say "Read HANDOVER.md and continue". Rewrite at every milestone (<= 3 KB). Mirko reads little: plain terms (B = baseline only; tiers 1 / 2 / 3). **D57 + D61: this (Spurmitte lane) chat has priority.** GPU only in his windows (nights 22-07, Mon-Fri 08:30-17:30). **D60: Mirko is away, work alone, ask nothing; target = ready Sunday morning ~07:00.** No TEST, push, mail, shutdown. Do not edit `autogt/` (GT chat owns it).

## Results of the night (cards experiments/E275_* .. E279_*; pages results/E275 .. E277/README.md)
- **E275 KEEP** hard close-node step (topo F1 +0.03, junction F1 +0.13, vertices per 100 m 28.9 -> 19.3; a rule step).
- **E276** arm t (tier loss) **TIE**, not adopted; arm s (sparse nodes) **failed its gate** (probability smears). CPU analyses of the eight notes in card 7a: long links persist only 64 - 65 % between frames, precision 0.17 - 0.19, still needed (-0.12 topo F1 without them); assignment fine; link thresholds a trade.
- **E277** the soft step trained inside the head (commit 66cc4084): reproduces the hard step (topo F1 0.621 / 0.624, junction F1 0.263 / 0.268, vertices 19) with no rule step left, but **CUT by the written guards** (seed 0: lanes found -0.039, F1@1.0 -0.015). Open for Mirko: are the hard-step guards right for a trained step.
- **E278** (diagnostic): the joint phase 7 raises topo F1 +0.03 - 0.05, lowers junction F1 under the step; stays.
- **E279 running** (card `experiments/E279_soft-close-node-step-r125/`): the same soft step at radius 1.25 m, same rule (`verdict277 --exp-var e279`, automatic). Jobs `E279.w_s0` (started 03:42), `w_s1` (waits for GPU memory < 8.5 GB: the BEV chat's smoke job runs), then `wq_s*` scoring; done ~06:30. BEV chat told (SendMessage) to hold its round until `temp/e276/CHAIN_DONE` exists again.

## The scheduler (`temp/e276/launch_chain.sh`, log `chain.log`, state `bash temp/e276/status.sh`; restart `bash temp/e276/restart_chain.sh`: keeps no state besides STARTED_* flags; edit its scripts only with it killed)
## When E279 is done (morning)
Verdict in `temp/e276/verdicts.log`; ledger notes (`temp/e276/ledger_notes_night3.sh` pattern), E279 card section 7, `results/E279/README.md` (+ `.gitignore` line), rebuild the report (`bash temp/e276/run_report.sh`; page `results/E276.report/index.html`), pictures `results/E277/figures/` (`temp/e276/run_views.sh`), artifact update (https://claude.ai/artifact/Vdey4Vj4D6CD942ymy1qwB: load artifact-design first), message BEV chat + GT chat (l5 merge may follow CHAIN_DONE), memory note. Nothing is pushed.
