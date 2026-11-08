# HANDOVER - 2026-11-08 17:35

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
**E280 (learned radius per node pair) and E281 (memory for long links) are CLOSED by their own CPU gates before any code (cards, section 3a; scratch tags `lcr/e280-scratch-1`, `lcr/e281-scratch-1`).** E280: pair features add <= +0.011 AUC over e_i, e_j; an oracle sparing real close pairs is worse than the fixed 1.25 m radius = the junction gain is the thinning of real nodes. E281: the previous frame adds <= +0.025 AUC (needed +0.05); hand fusions lift long-link persistence 0.64 -> 0.72 - 0.76 only by losing 0.02 - 0.04 lanes found. **Next, Mirko's pick:** matching-based node loss; a 20 min CPU check of a LOWER long-link threshold (E281 3a); or the open 1.25 m decision. He declined a third E279 seed. Nothing pushed; scratch tag `lcr/e276-scratch-2`.
