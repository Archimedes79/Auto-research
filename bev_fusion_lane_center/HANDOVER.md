# HANDOVER - 2026-11-08 16:20

New chat: start in the repository folder, say "Read HANDOVER.md and continue". Rewrite at every milestone (<= 3 KB). Mirko reads little: plain terms (B = baseline only; tiers 1 / 2 / 3). **D57 + D61: this (Spurmitte lane) chat has priority.** GPU only in his windows (nights 22-07, Mon-Fri 08:30-17:30). **D60: Mirko is away, work alone; target = ready Sunday morning.** No TEST, push, mail, shutdown. Do not edit `autogt/` (GT chat owns it).

## Results of the night (cards experiments/E275_* .. E279_*; pages results/E275 .. E279/README.md; report page results/E276.report/index.html, artifact https://claude.ai/artifact/Vdey4Vj4D6CD942ymy1qwB v7)
- **E275 KEEP** hard close-node step (topo F1 +0.03, junction F1 +0.13, vertices 28.9 -> 19.3; a rule step).
- **E276** arm t (tier loss) **TIE**, not adopted; arm s (sparse nodes) **failed its gate**. CPU analyses of the eight notes (card 7a): long links persist only 64 - 65 % between frames, precision 0.17 - 0.19, still needed; assignment fine; link thresholds a trade.
- **E277** soft step trained in the head, radius 1.5 m: reproduces the hard step but **CUT by the guards** (seed 0 lanes found -0.039). **E279 radius 1.25 m: KEEP** (verdict277 06:06): topo F1 0.618 / 0.620, junction F1 0.223 / 0.218, lanes found 0.618 / 0.632, vertices 21; all guards held, thin margins on seed 0 (topo gain +0.0210 vs band 0.02). The thinning is learned by the network (D55 for this step).
- **E278** diagnostic: joint phase 7 raises topo F1, lowers junction F1; stays.
- **Open for Mirko (no D-id opened):** adopt the 1.25 m soft step as the delivered thinning in place of the hard read-out (changes baseline B, D58); guard values for a trained step.

## State of the machine
Lane chain done (`temp/e276/CHAIN_DONE`). **GPU: BEV chat's t23.C_s0 runs until about 18:15 (it stops on my word, D57); no lane GPU job without telling it.** GT chat merged l5 into live autogt/ at ~06:40 (`lane_gt` defaults still give the l4 store 0702a065c6); it commits + tags (`autogt/lane-l5`, `autogt/v12`) after its live checks: wait for its 'live checks done' before a new lane job. WSL page-cache fragmentation killed two E279 runs at 04:18 / 04:32 (`temp/e276/reclaim.py` frees it, resume works).

## If continuing
**E280 (learned radius per node pair) is CLOSED, 16:20, by its own gate before any code (card section 3a, CPU, both seeds):** the extra pair features add at most +0.011 AUC over the head's own e_i, e_j (needed +0.05); an oracle sparing the real close pairs reaches junction F1 0.193 / 0.178 (needed 0.255 / 0.241) and is worse than the fixed 1.25 m radius = the junction gain is the thinning of real nodes. **Next, Mirko's pick:** matching-based node loss (own card + pre-check) or memory for long links; he declined a third E279 seed. Rebuild the report: `wsl bash temp/e276/run_report.sh`; pictures: `temp/e276/run_views.sh`. Nothing is pushed; scratch tag `lcr/e276-scratch-2` (refresh with `temp/e276/tag_scratch2.sh` under a new name).
