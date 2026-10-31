# E259 Is the spur / bubble clean-up of the assembly needed?

## 0. What the tools run (the fields of `experiment.json`)
| field | value |
|---|---|
| id / card / title | E259 / H21.S2 / clean-up ablation (simpler and faster assembly) |
| kind | variant (post-processing of cached raw outputs, no training; a script, no metric cards) |
| metric, direction, band | Q, higher, 0.010; guards J, F1@0.5, vert100, false length |
| prediction | Q change -0.001 (tolerance 0.003); assembly time -0.9 ms relative |
| rule | see section 1 |
| budget | CPU about 25 minutes, 1 thread |

## 1. Card
| | |
|---|---|
| lens | simplification, speed |
| group | A3 sparse lane graph (assembly) |
| sources | E258 profile (`simplify` 11 % of the assembly after relocation 21 %, anchoring 16 %, union 16 %), E217 probe (clean-up variants on the champion: scores printed, no record of a clear effect), E257 (the DP output stage already removes vertices; spurs and bubbles are different: dangling stubs and tiny loops) |
| status | proposed |

- **Statement**: with fork pruning (K190) in place the spur (< 5 m dangling branch) and bubble (< 2.5 m loop) removal changes no metric beyond the noise, so it can be dropped: one stage and two parameters less, about 0.9 ms per frame (11 % of the assembly).
- **Mechanism**: spurs and bubbles are artefacts of the path union (a node snapped next to an existing edge); their length is small, so false-length and F1 barely react; fork pruning already removes the unsupported forks that most spurs belong to.
- **Prediction** {value -0.001, tolerance 0.003, class "removing a post-processing step"}: Q -0.001, J 0.000 to -0.005, false length +0.002; the E217 probe printed nothing that contradicts it.
- **Decision rule** (json): all seven models (fast_recall, champion, recall, e24_j1rc_s1, s2, s3, and fast_recall_na), with the output stage off and on (dp_eps 0.2): Q loss <= 0.003, J loss <= 0.01, F1@0.5 loss <= 0.005, vert100 change <= 0.3, false length rise <= 0.01, saving >= 0.5 ms (interleaved A/B, relative, loaded machine). A single model outside the guards keeps the stage. TIE inside the guards with saving below 0.5 ms keeps the stage (nothing is gained by a change).
- **Test design**: layer 1 the base reproduces the stored card of each model (Q within 5e-4, vert100 within 0.01); the same raw outputs for both variants; timing 5 repetitions of 200 frames interleaved, median of medians, load average printed; the absolute saving is repeated in an idle window.
- **Cost / kill**: CPU 25 minutes; none.
- **Inputs**: ready (`temp/loop20/e255_matched.py` loader, `temp/loop21/e257_simplify.py` simplify_dp now in `p16_infer`).
- **Referee**: not run (CPU analysis of existing outputs); a code change goes to an `ar-reviewer` pass.

## 3. Summary (2026-10-30 / 31, runs E259.a0 (ablation) and E259.b0 (CPU time); VAL 873, cached raw outputs, CPU)
- **Decision by the written rule: the spur / bubble clean-up STAYS.** On all seven models (fast_recall, champion, recall, j1rc seeds 1 - 3, fast_recall_na) switching it off raises vert100 by +0.44 .. +0.73 (guard <= 0.3 fails on every model, with and without the output stage), false length by +0.002 .. +0.003 and lowers topo F1 by 0.005 - 0.006. What it buys back: Q +0.0001 .. +0.0027 and lanes found +0.017 .. +0.021 (the clean-up cuts short real lane stubs), J -0.004 .. +0.006. At a matched vertex budget (+0.5 vertices is worth about -0.005 Q at the E236 slope) the Q gain disappears: no free lunch either way.
- **Prediction** {Q -0.001, tolerance 0.003, saving 0.9 ms}: Q change +0.0015 mean (reviewer's recomputation) (within tolerance, wrong sign); the saving could not be measured (below).
- **Timing, honestly (framework item N18):** the machine was oversubscribed during both runs (load average 4 - 15; the other chat's lane plus more). Wall-clock paired savings ranged from -6.9 to +5.6 ms (E259.a0). The CPU time of the process (`process_time`, E259.b0) did not help: it equals the wall time (the process is not descheduled; the core itself runs slower under SMT / cache contention), repetition ranges of one variant span 15 - 35 ms. Only effects of about 25 % of the assembly show above the noise: junction anchoring off saves about 5 / 4 ms of 19.6 / 17.5 (medians 5.24 / 4.29; provisional, load 4 - 15) (positive in 10 of 10 repetitions on fast_recall and champion), fork relocation off 5.4 / 3.4 ms (positive in 7 of 10), the clean-up off nothing readable (range contains zero), the output stage dp 0.2 COSTS about 0.7 - 1.5 ms loaded (negative in 6 of 10: a python loop over polylines). Timing values are provisional until an idle window; the stage ranking is consistent with E258.
- **Consequences:** (1) no stage removal beyond E258's anchoring candidate for the j1 family; (2) the DP output stage has a real cost (about 5 % of the assembly): worth a vectorised version only if it stays in the product; (3) the framework needs a rule: no timing claim on a machine whose load average exceeds the number of physical cores; timing is requested from the owner of the machine (N1 / D19).
- **Weaknesses:** a script (no metric cards), same raw outputs only, seven models of three builders; the timing is not usable for absolute numbers.
