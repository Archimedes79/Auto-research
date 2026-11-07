# HANDOVER - 2026-11-07 20:20

New chat: start in the repository folder, say "Read HANDOVER.md and continue". Rewrite at every milestone (<= 3 KB). Mirko reads little: plain terms (B = baseline only; tiers 1 / 2 / 3). **D57: Spurmitte first.** GPU only in his windows (nights 22-07, Mon-Fri 08:30-17:30). **D60 (11-07): Mirko is away, work alone, ask nothing**; no TEST, push, mail, shutdown.

## Where we are
**Baseline B (D58)** = E273 head (`runs/e273_p_s0/s1`): VAL 873 topo F1 0.596, junction F1 0.125 (P 0.07), `vert100_dp` 28.9, assignment lane_acc 0.98 per node. E274 closed (CUT).
**E275** (close-node suppression): scoring attempt 1 died 11-05 (path bug, fixed). `temp/e275/launch_scoring.sh` is armed, starts 22:00 Sat, 4 runs ~75 min, flag `temp/e275/SCORED`; it now reads the lane GT **l4** (geometry identical to l3, checked on all 873 VAL frames; its `--repro` gate re-checks the numbers). After: `analyse.py`, `verdict275`, ledger notes, card section 7, `results/E275/README.md`.
**E276** (card `experiments/E276_gt-tiers-sparse-nodes-tier-loss/`, D59 / D60): **autogt lane GT l4 is merged and committed** (40fe02b7, tag `autogt/lane-l4`, signature 0702a065c6, release v11; critic pass done, 4 blocking fixes; selftests green). Full store `data/autogt/lane/0702a065c6` still generating from `temp/gt_l4` (34,149 files, ETA ~21:05, flag `temp/gt_l4/GEN_DONE`; do not touch the signature files of the copy). Head code, tests, card committed (8497efe4, 6485f788). CPU pre-experiments P1-P4 are in card section 7a (sparse node target keeps the oracle geometry with 56 % fewer nodes; `vert100_dp` of a perfect network is 11.4 under any node target; B per tier: tier 1 recall 0.879 / precision 0.898, tier 3 0.815 / 0.882; extension rule recall 0.92-0.95). Rule of arm s now judges delivered nodes per 100 m.
**Chain** `temp/e276/launch_chain.sh` (window-aware, co-runs with E275 on the second GPU slot): S0 GPU smoke of arm t / s, A score B on l4, B train t x2 then score, C train s x2 then score, `verdict276`. **Not started yet:** needs the full store, then `touch temp/e276/READY`, then `nohup bash temp/e276/launch_chain.sh` (WSL). The BEV chat holds its t23 chain while READY exists and CHAIN_DONE does not (agreed); tell it when READY exists. Cron 23:43 holds the rest of the plan.

## After the store is complete
1. `check_geometry.py` (l3 vs l4, all VAL + 500) and `exp_tier_stats.py` on the full store; resume `lane_gt.generate` from the live source to refresh the index (lane_version l4, commit stamp); 2. tests (`temp/e276/run_tests.sh`); 3. READY + chain + message to the BEV chat; 4. dead-code / bug report of the background agent: review, fix what is safe, commit.

## Where
Report https://claude.ai/artifact/Vdey4Vj4D6CD942ymy1qwB (v5, lacks E274-E276). Scratch tags `lcr/e275-scratch`, `lcr/e276-scratch`. Commit by path (`autoresearch/tools/commit_paths.py`). The stale draft store `data/autogt/lane/c96c18fb1f` can go via `package.py`.
