# HANDOVER - 2026-11-08 03:30

New chat: start in the repository folder, say "Read HANDOVER.md and continue". Rewrite at every milestone (<= 3 KB). Mirko reads little: plain terms (B = baseline only; tiers 1 / 2 / 3). **D57 + D61: this (Spurmitte lane) chat has priority.** GPU only in his windows (nights 22-07, Mon-Fri 08:30-17:30). **D60: Mirko is away, work alone, ask nothing; target = ready Sunday morning ~07:00.** No TEST, push, mail, shutdown. Do not edit `autogt/` (GT chat owns it; GT wishes by message).

## Results of the night so far (cards: experiments/E275_*, E276_*, E277_*; pages results/E275/README.md)
- **E275 closed (47f5e6a0): KEEP** the hard close-node step (topo F1 +0.03, junction F1 +0.13, vertices per 100 m 28.9 -> 19.3; a rule step, D55 stop-gap).
- **E276: arm t (tier loss) = TIE, not adopted** (own-lane gain -0.002 / +0.010; far-lane recall -0.10, lanes found -0.09). **Arm s (sparse nodes) failed its phase-3 gate** (CUT): the head cannot place nodes on a sparse target (probability smears; diag in card 7b).
- **CPU analyses of the eight notes** (card 7a): P5b long links persist only 64 - 65 % (GT 98 %); P8 / P8b assignment is fine (97.6 % per node, 88 % of map lanes claimed / shared; my first "45 % uncovered" was a metric artefact); P9 link probability over-confident, long links precision 0.17 - 0.19; P10 link thresholds are a trade, no gain, long links indispensable (without them topo F1 -0.12).
- **E278 diagnostic** (phase-6 model scored, no card): the joint phase 7 raises topo F1 (+0.05) and long-link precision, lowers duplicates; stays. Seed 1 (`E278.qp_s1`) in the queue.
- **E277 = the soft close-node step in the head, trained** (code + tests + critic pass 66cc4084; card `experiments/E277_soft-close-node-step/`): `E277.u_s0` / `u_s1` DONE (03:15 / 03:20; live 150-frame gate: topo F1 0.619 / 0.623, junction F1 0.27 / 0.265, vertices per 100 m 19.0 / 18.8). `uq_s0` scoring since 03:21, `uq_s1` queued, then `verdict277` automatic (`temp/e276/verdicts.log`; KEEP = replaces the hard step). Their phase-3 gate read the raw head (found by the critic after the start; cards judge).

## The scheduler (`temp/e276/launch_chain.sh`, log `temp/e276/chain.log`, state `bash temp/e276/status.sh`; restart `bash temp/e276/restart_chain.sh`, safe: it keeps no state besides the STARTED_* flags)
Jobs left: qp1, uq0 (running), uq1; CHAIN_DONE when all done. BEV chat holds t23 while READY exists and CHAIN_DONE does not: message it (and the GT chat) at CHAIN_DONE.

## After the jobs (morning)
Comparison pictures (B | B + hard step | u | arm t, six frames) are rendered into `results/E277/figures/` by `temp/e276/run_views.sh` (flag VIEWS_DONE).
`temp/e276/compare_views.py` (pictures B | B + hard step | u of the same frames, `wsl bash temp/e276/run_views_test.sh` pattern), `temp/e276/e276_build_report.py` via `run_report.sh` (page `results/E276.report/index.html`, reads result files only), artifact update (https://claude.ai/artifact/Vdey4Vj4D6CD942ymy1qwB: load artifact-design first), E277 card section 7 + `results/E277/README.md` (+ `.gitignore` line) + ledger notes of the E277 / E278 rows, messages, memory note. Nothing is pushed.
