# HANDOVER - 2026-11-08 10:00

New chat: start in the repository folder, say "Read HANDOVER.md and continue". Rewrite at every milestone (<= 3 KB). Mirko reads little: plain terms (B = baseline only; tiers 1 / 2 / 3). **D57 + D61: this (Spurmitte lane) chat has priority.** GPU only in his windows (nights 22-07, Mon-Fri 08:30-17:30). **D60: Mirko is away, work alone; target = ready Sunday morning.** No TEST, push, mail, shutdown. Do not edit `autogt/` (GT chat owns it).

## Results of the night (cards experiments/E275_* .. E279_*; pages results/E275 .. E279/README.md; report page results/E276.report/index.html, artifact https://claude.ai/artifact/Vdey4Vj4D6CD942ymy1qwB v7)
- **E275 KEEP** hard close-node step (topo F1 +0.03, junction F1 +0.13, vertices 28.9 -> 19.3; a rule step).
- **E276** arm t (tier loss) **TIE**, not adopted; arm s (sparse nodes) **failed its gate**. CPU analyses of the eight notes (card 7a): long links persist only 64 - 65 % between frames, precision 0.17 - 0.19, still needed; assignment fine; link thresholds a trade.
- **E277** soft step trained in the head, radius 1.5 m: reproduces the hard step but **CUT by the guards** (seed 0 lanes found -0.039). **E279 radius 1.25 m: KEEP** (verdict277 06:06): topo F1 0.618 / 0.620, junction F1 0.223 / 0.218, lanes found 0.618 / 0.632, vertices 21; all guards held, thin margins on seed 0 (topo gain +0.0210 vs band 0.02). The thinning is learned by the network (D55 for this step).
- **E278** diagnostic: joint phase 7 raises topo F1, lowers junction F1; stays.
- **Open for Mirko (no D-id opened):** adopt the 1.25 m soft step as the delivered thinning in place of the hard read-out (changes baseline B, D58); guard values for a trained step.

## State of the machine
Lane chain done: `temp/e276/CHAIN_DONE` exists (BEV chat's chain5 holds on it; told). GPU free. GT chat merged l5 into live autogt/ at ~06:40 (code signature 2464b5755f; `lane_gt` defaults still give the l4 store 0702a065c6, lane graphs byte-equal; new: `load(token, rig="bev_fusion")`); it commits + tags (`autogt/lane-l5`, `autogt/v12`) after its live checks: wait for its 'live checks done' before a new lane job. Scheduler exited; memory watchdog stopped. WSL page-cache fragmentation killed two E279 runs at 04:18 / 04:32 (ENOMEM in DataLoader workers: `temp/e276/reclaim.py` frees it, resume from ckpt works).

## If continuing
**Next: card E280** (learned radius per node pair; revised after a critic pass; nothing coded or run): step 1 = CPU pre-check P11 (features + oracle gate, card section 3); only if it passes: code, tests, critic, GPU in a later window (tonight: BEV t23, Image Encoder B2). Other levers: memory for long links, matching-based node loss; Mirko declined a third E279 seed for now. Rebuild the report: `wsl bash temp/e276/run_report.sh`; pictures: `temp/e276/run_views.sh`. Nothing is pushed; scratch tag `lcr/e276-scratch-2` (refresh with `temp/e276/tag_scratch2.sh` under a new name).
