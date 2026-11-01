# PLAN - experiment tree of the current loop (loop 22)

Written in P2 by ar-planner (2026-11-01 ~01:15), stable during the loop: only the status column changes. The schedule (what runs next) is `schedule.csv` (`schedule.py show | next`). Loop 21's plan: `history/plans/PLAN_loop21.md`.

Question of the loop: which change of the existing fast chain closes most of gap 1 (coverage at range: `recall` lanes found 0.767 / R@1 far 0.771 against the control's 0.800 / 0.811) at Q >= 0.735, <= 14.3 vertices per 100 m (per-piece Douglas-Peucker 0.3 m, the oracle basis of the budget, E257 section 5 / D25) and recall-chain speed, and which system is the D28 prototype that gets the one TEST check. Prompt: `history/prompts/PROGRAM_loop22.md` (STOP_AT 06:30, +2 - 3 h for jobs in flight, then shutdown; one TEST check; no mail).

Levels this loop owns: level 3 (operating points and inference / assembly settings of existing checkpoints: E261, E265, E266, E267) and one level-2 node (output ensemble of two seeds, E264); one confirmation training (E262). No new loss, no new architecture (E263 deferred).

Budget: CPU now - 05:45, at most 2 jobs at a time, 2 threads, ~2 GB each (a bev_fusion training holds the box); lane GPU 05:45 - ~08:00 (~100 GPU-min, both slots, timing first and exclusive); code by an ar-coder (in this order: S1 cache key, S2 ring + ring metric, S3 union kind + two-member timing), reviewed, planned as ready 05:30. Comparisons that decide: E261, E262, E264, E265, E266, E267 (six, each by its own written rule); one TEST use (E268, reserved).

## Tree
Every node: card (section 1, referee paragraph) -> detail plan (section 2) -> runs through `run.py` -> `decide.py --experiment E<id> --write` plus the lead checks named in the card's experiment.json notes -> summary. VAL 873, cached raw outputs unless stated; every number labelled "VAL, layer 3" (or "VAL, selection-CV, layer 3" where the operating point was chosen on the same frames).

| node | experiment | hypothesis (short) | decision rule (short) | cost | value x P / cost | priority | status |
|---|---|---|---|---|---|---|---|
| N0 | gap 1 (root) | coverage at range is lost in the chain, not in the features (E265 step 0: valid tree 0.866 -> output 0.771 R@1 far) | - | - | - | - | open |
| N1 | E265 spurs kept (`spur_len 0`) | the spur half of the clean-up deletes true far geometry; on the DP budget the spur-kept graphs fit | KEEP if mean lanes found >= +0.010 (band) and R@1 far >= +0.004 on 4 replication checkpoints, guards on each (Q -0.003, dup +0.005, FL +0.005, DP <= 14.3) and above the equal-budget tau frontier; TIE with guards = cheaper = spurs kept too | CPU ~1.7 h, 0 GPU | +0.012 x 0.6 / small: highest | 1 | **KEEP** (04:05: lanes found +0.0115, R@1 far +0.0065, Q +0.0039, 4 / 4 intervals, DP <= 13.83, above the frontier via Q; candidate S = `recall_sp0_dp`) |
| N2 | E261 rc2 at its own operating point | weight 6 buys coverage that rc1's threshold cannot | cheap rung first: CUT if rc1's best admissible point (tau down to 0.1) is within 0.010 lanes found of rc2's 0.790; else 20-split CV: PASS if held-out lanes found +0.015, R@1 far +0.010, +0.010 over the threshold control, guards | CPU ~1.5 h, 0 GPU | +0.013 x 0.45 / small | 1 | **CUT** (04:40: 22-scene card lanes found +0.0133 < +0.015, R@1 far +0.0054; held-out DP > 14.3 in 10 / 13 halves, Q -0.0075); W out |
| N2a | E262 rc2 seed twin | the rc2 effect is the recipe, not seed 0 | 2 seeds vs rc1 / rc1s1: lanes found +0.015, R@1 far +0.010, dup +0.030 max | GPU 60 min | confirmation only | 4 (first to drop) | cancelled (E261 CUT) |
| N3 | E266 seed ring at +-47.5 m | the far gap is the 45 - 48 m ring the inference lattice does not reach | G0 flag changes output, G1 tree ring +0.02; KEEP if mean R@1 far >= +0.010 (band) on 4 checkpoints, recall alone >= +0.010 with interval > 0, inner-far guard +-0.005, Q / topo / FL / dup guards | GPU ~25 min + CPU 0.5 h + code S1, S2 | +0.012 x 0.5 / medium | 2 | **KEEP** (06:10: mean R@1 far +0.0152, 4 / 4, Q -0.0016; G0 numeric bar missed 1.88 x, flag checked; timing missing; candidate R) |
| N4 | E267 stack 2 x 2 (spurs x ring) on recall, recall_s1 | the two act on different losses and add | KEEP sr if lanes found and R@1 far >= +0.015 vs c on both seeds, the ring adds >= +0.005 R@1 far over s and the spurs >= +0.010 lanes found over r (headline: held to the band), guards | CPU ~0.5 h after N3's caches | decides the prototype setting | 2 | **CUT** (06:10: coverage clauses hold, lanes found +0.022, R@1 far +0.027; false length +0.006 and P@1 far -0.015 miss their guards; vs S only +0.009) |
| N5 | E264 union of two j1rc seeds | seed-specific far hypotheses: the union of two trees finds more lanes | KEEP if u23 lanes found >= +0.030, R@1 far >= +0.025 over s3, +0.010 over the matched-false-length s3, dup +0.020, FL <= 0.062 abs, DP <= 14.3, same signs on u01, ms <= 30 idle | GPU 15 min (timing) + CPU ~1 h + code S1, S3 | +0.045 x 0.25 (speed risk) / medium | 3 | **not adopted** (05:20: decide.py KEEP on its clauses, lanes found +0.047, replicated, +0.038 over matched FL; but FL 0.0635 > 0.062 and no idle timing; next loop: wider vertex merge + timing) |
| N6 | E263 seed soup | - | - | - | referee: not tonight (majority 0.653 < single 0.681) | - | deferred (revise) |
| N7 | E268 TEST check (reserved id) | the prototype's VAL gain over `recall` survives on TEST against the control | report only (no selection on TEST): prototype vs control on 833 frames, paired bootstrap; the E230.rc TEST row of rc1 + pruning 0.4 is the in-sample-free reference | GPU 5 - 10 min | the D28 deliverable | after P | **done** (06:13, TEST 833: lanes found 0.740 (prediction 0.735 - 0.74), Q 0.727, R@1 far 0.749; vs control lanes found -0.021, Q +0.015; vs E230.rc +0.016 lanes found) |

Order (machine time; the schedule holds the commands):
- 01:30 - 05:30 CPU, two at a time: E261.g (rung, ~10 min) and E265.c (layer-1 control re-score, ~15 min) start now; then E265.a, E265.v, E265.b, E265.eb; E261.cv once its script (lead) exists and the rung did not kill it; E261.ev; decisions E265 (~03:30) and E261 (~04:30); `recall_sp0_dp` packaged if E265 adopts spurs (the 06:30 fallback-plus).
- 05:30 code landed and reviewed: E266.gold (golden check + `run_all_tests.sh`), registry entries of ring / union names (lead), E264.u (u23 first) on the CPU until 05:45.
- 05:45 - 06:00 GPU exclusive, CPU quiet: E264.t idle timing (union of two j1rc members, single j1rc, recall as references).
- 06:00 GPU slot 1: E266.b0 (recall ring) -> G0 / G1 (CPU) -> b1 (recall_s1) -> b2, b3 (~06:35). Slot 2: E262.a1 (if E261 PASS) 06:00 - ~07:00, then E262.ev.
- 06:00 - 06:45 CPU: E266.c, E264 rest (u01, u4, urc, DP, matched-FL, matched-vertex), E267.s, E267.sr (after b1), decisions E264, E266, E267.
- 06:30 STOP_AT: the prototype of record is the best candidate decided by then (rule below); in-flight E262 / E267 may replace it until ~07:15.
- ~07:15 E268 TEST check (GPU 5 - 10 min), then P4: summaries, `check_numbers.py`, ar-reviewer (numbers), `check_structure.py`, HANDOVER, shutdown ~08:30 when nothing of ours runs.
If the GPU window starts late or shrinks: drop E262 first (rc2 then stays a one-seed, VAL-selected candidate, labelled so), then E266.b2 / b3 (keep b0 / b1, the stack needs them), then E264.t (E264 then cannot be the prototype: "quality KEEP, speed pending" is not deliverable).

## The prototype (D28): from tonight's verdicts to the registry entry that gets the TEST check
Written before any number of tonight is read. Fallback, already packaged: `recall_dp` (`tools/prototype/lane_prototype.py`, rc1 + pruning 0.4 + output stage dp_eps 0.2).

Candidates, each admitted only through its experiment's verdict, and only if on VAL 873 it holds the joint guards: Q >= 0.735, per-piece DP 0.3 m <= 14.3, false length <= 0.062, dup <= `recall` + 0.030, topo F1 >= `recall` - 0.010.
| candidate | made of | admitted when | evidence |
|---|---|---|---|
| F | `recall_dp` | always | 2 seeds (recall / recall_s1), TEST once (E230.rc) |
| S | recall + `spur_len 0` | E265 KEEP, or TIE with every guard holding (the cheaper variant) | 4 replication checkpoints |
| R | recall + ring | E266 KEEP | 4 checkpoints |
| SR | recall + ring + `spur_len 0` | E267 KEEP | 2 seeds of rc1 |
| W | rc2 at the E261 point (`recall6`) | E261 PASS; with E262 KEEP two seeds, without E262 "one seed, VAL-selected" | selection-CV; not stackable with S / R (no vertex headroom, referee) |
| U | union of j1rc s2 + s3 (`fast_recall_ens2`) | E264 KEEP including ms <= 30 on the idle GPU | held-out pair + replication pair |

Choice: the admitted candidate with the highest lanes found; a candidate replaces a simpler one only if it is at least +0.010 lanes found better (the band) with R@1 far not lower; inside the band the simpler one stays, in this order of simplicity: F < S < R < SR < W < U (assembly-only before inference setting before another checkpoint before a second forward). W enters only with E262 KEEP if the TEST run waits for it; if E262 is not decided by 07:30, W can still be the prototype but is labelled one-seed. Expected: SR or S (about lanes found 0.78, R@1 far 0.79).
Packaging: the winner becomes registry entry `<name>_dp` (dp_eps 0.2, node-id count next to the DP count), `lane_prototype.py <name>_dp` must reproduce its VAL card (layer 1), video of two median scenes; the summary page names the rule steps of the delivered graph (path union, clean-up, fork pruning: the E2E gap) and what is unchecked (route success on spur-kept graphs, failure series).
TEST (E268): one run `evaluate.py <name>_dp --split test --test-id E268.p --compare control` after the row in `evaluation/TEST_USAGE_LOG.md` (protected: grant under D28) names E268.p; nothing is selected or tuned on it; reported next to the E230 TEST rows (rc1 + pruning 0.4: Q 0.727, lanes found 0.724; control: lanes found 0.762).

## System improvements
| item | lack in the software | change | done when | status |
|---|---|---|---|---|
| S1 | `evaluate.builder_raw` keys the raw cache by run / M / K0 only (and the loop-18 fallback by run / M), so an entry that differs in any other `set` field silently re-scores another entry's outputs (E263 / E264 / E266 referees) | key = sorted extra `set` items in the file name; old names valid when `set` has only M / K0; union entries never resolve through a member's key | spec test: two entries differing only in `lattice_ring` give two files, existing entries their existing files; `evaluate.py --golden check --cached-only` 6 of 6; ar-reviewer (code) | coder, by 05:30 |
| S2 | the inference lattice stops at +-45 m; no ring metric | `DefaultCfg.lattice_ring` / `lattice_ext` in `p16_builder.seed_candidates` (default bit-identical); `topo_metrics.range_counts` gains `R@1_far_inner` / `R@1_far_ring` (and P) | spec tests (default tensor equal, ring adds 80 points at 47.5 m); the ring metric reproduces `step0_ring.txt` (recall 0.8014 / 0.7016); reviewer | coder, by 05:30 |
| S3 | no output ensemble | `p16_infer.union_raw` + registry kind `union` (`members`) in `evaluate.graphs_of`; `p18_latency.py` option for two members | identity union reproduces its member's graph hash on 873 frames; default path untouched; reviewer | coder, by 05:30 |
| S4 | the vertex budget basis (per-piece DP 0.3 m) is not in the metric cards, so no rule can read it | `vert100_dp` in `sparse_metrics.card_any(extra=True)` | a card carries it; after tonight (tonight the lead reads it from `cleanup_split.py` / `frontier_scratch.py` / `e261_cv.py`) | open, after the loop |
| S5 | `run.py`'s snapshot covers `code_dirs` only; the step-0 scripts the CPU arms call live in `experiments/E261_*` / `E265_*` (untracked) | commit the six experiment folders by path before the first run | the run's commit contains the script (`git show lcr/E261.g --stat`) | lead, before 01:45 |
| S6 | `decide.py` judges a clause against a non-control arm without an interval and flags it "one seed" | (tools chat) take the interval of a card that compared against that arm | the E261 / E264 threshold-control clauses can live in experiment.json | open; tonight they are lead checks |

## Controls and statistics
- Matched reference: every variant card is written with `--compare <its control>` (same frames, paired scene bootstrap); the control cards of the decisions are re-scored tonight with the current code (tags `l22` / `l22r`, layer 1 through `--expect` of the stored numbers), so every card of a comparison carries the same protocol field.
- Null / built-in nulls: arm c on the same frames; E265 placebo (`bubble_r 0`); E266 inner-far guard (seeds inside +-45 m can be displaced: a guard, not a null); E264 identity union and the no-diversity union of rc1 + rc1s1.
- Equal budget: E265 tau frontier to 0.1 (rule for an unreachable DP in E265 section 2); E261 threshold-only control t to tau 0.1; E264 matched-false-length and matched-vertex single models.
- Seed twins: recall_s1 everywhere; j1rc s1 - s3; E262 for rc2.
- Band 0.010 on every headline; a KEEP bar below it is raised to it (decide.py does the same); deterministic arms (one checkpoint, a setting) are judged by the paired scene-bootstrap interval.
- Selection on VAL only; operating points chosen on the 873 frames are "VAL, selection-CV" (E261) or in-sample (E264's val250 choices judged on 873 overlap 250 frames: labelled); TEST once (E268), after the prototype is fixed.
- Speed: ms only from the idle-GPU session (E264.t); assembly-only and ring changes are "same network, assembly relative" until then.

## End of loop (P4)
- per node: result and decision (one line each, numbers from metric cards)
- best system now (the prototype, its TEST line), and what changed against `recall_dp`
- questions closed (E261 CUT would add a third answer to "larger recall weights"), open points
- decisions for Mirko: a default change of the recall family (spurs / ring) touches `champion` only by a D-entry; D25, D4 stay open
- checks: `check_numbers.py`, ar-reviewer (numbers), `check_structure.py`, `run_all_tests.sh`, `clean_temp.py loop22` dry run

### Filled 2026-11-01 ~06:40 (lead)
- Per node (VAL 873 unless named; cards and run logs in the experiment files, section 3 each): N1 E265 spurs kept **KEEP** (lanes found +0.0115, R@1 far +0.0065, Q +0.0039 on 4 replication checkpoints); N2 E261 rc2 **CUT** (+0.0133 in-sample, over the vertex budget held out); N2a E262 cancelled; N3 E266 seed ring **KEEP with two rule deviations** (R@1 far +0.0152 mean, Q -0.0016; G0 kill criterion overridden after a debug round; inner guard exceeded on recall_s1 by a gain +0.0068); N4 E267 ring + spurs **CUT** (lanes found +0.022, R@1 far +0.027, but false length +0.006 and P@1 far -0.015); N5 E264 two-seed union **not adopted** (lanes found +0.047 replicated, +0.038 over matched false length; false length 0.0635 > 0.062, no idle timing); N6 E263 deferred; N7 E268 TEST done.
- Best system now: `recall_sp0_dp` (= `recall_dp` + spurs kept). VAL: lanes found 0.767 -> 0.781, R@1 far 0.771 -> 0.780, Q 0.740 -> 0.742. TEST (E268.p, 833 frames): Q 0.727, lanes found 0.740, R@1 far 0.749; control lanes found 0.762, Q 0.712; +0.016 lanes found over E230.rc at equal Q. Packaged: `tools/prototype/lane_prototype.py recall_sp0_dp` reproduces its card, video in `runs/prototype/recall_sp0_dp_video/`.
- Questions closed: "larger recall weight buys coverage within budget" answered no a third time (loop 19 weight 3, E261 weight 6 at its own point): goes to closed_questions.md with the next loop's filing. Open: precision at range for the stack and the union (both coverage-positive, both stop at a precision guard); speed of the ring (441 candidates) and of the union (two forwards) on the idle protocol; route success on spur-kept graphs.
- Decisions for Mirko: a default change of the recall family (spurs kept, ring) and of `champion` needs a D-entry; D25, D4 stay open.
- Checks done: check_numbers on the morning page (81 / 81 found); check_structure (loop-22 dead paths fixed; the rest older or other chats' rows); golden + run_all_tests PASS (E266.gold, 05:00); ar-reviewer (numbers) on both morning pages 06:45: no wrong table value; 8 + 5 label / rule findings fixed (E266 deviations, E265 frontier via the Q branch only, E230.rc another code version, timing labels); `clean_temp.py loop22` not run (the loop-22 records point into temp/loop22/).
