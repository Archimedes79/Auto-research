# E258 Where the assembly time goes, and a saving of at least 1 ms with identical graphs

## 0. What the tools run (the fields of `experiment.json`)
| field | value |
|---|---|
| id / card / title | E258 / H21.S1 / profile of `assemble()` and one targeted speed-up |
| kind | variant (code change of one stage, same output) |
| metric, direction, band | asm_ms (ms per frame, CPU assembly), lower, 0.5 |
| prediction | -1.0 ms (tolerance 1.0) |
| rule | see section 1 |
| budget | CPU, about 30 minutes, 1 thread |

## 1. Card
| | |
|---|---|
| lens | speed, system review |
| group | A3 sparse lane graph (assembly), cross-cutting speed |
| sources | E253 (valid session 2026-10-27: chain 18.5 - 18.8 ms without recursion, assembly 7.5 - 7.8 ms of it, 8.8 - 8.9 ms for the recall variants), loop 18 (K191 numba union, 873 of 873 identical), `training/pipeline/p16_infer.py` |
| status | proposed |

- **Statement**: after the numba union of loop 18 the assembly is still 40 % of the chain on the CPU; one stage dominates it, and a change of that stage (vectorise, cache, or skip a no-op) saves at least 1.0 ms per frame without changing a single graph.
- **Mechanism**: the assembly runs python loops over nodes and edges (path union, simplify spur / bubble removal, junction head, fork pruning). The numba kernels cover the union; the remaining stages are plain python and numpy with small arrays, where interpreter overhead, not arithmetic, costs the time.
- **Prediction** {value -1.0 ms, tolerance 1.0, class "targeted speed-up after a profile"}: the profile shows two stages above 15 % each; the change of the larger one saves 1.0 ms.
- **Decision rule**: step 0 is a diagnostic: stage timers (`time.perf_counter` around the stages in a copy) and cProfile on 200 frames; if no stage exceeds 15 % of the assembly the experiment ends as CUT (nothing to optimise). Step 1 only for a stage above 15 %: the change goes into a copy of the code under `temp/loop21/`, identical graphs on 873 of 873 frames (same comparison as `p18_asm_check.py`: vertices, edges, node ids), interleaved A/B timing, 5 repetitions, median of medians. KEEP if >= 1.0 ms faster; between 0.5 and 1.0 INCONCLUSIVE (repeat on an idle window); below 0.5 TIE.
- **Test design**: layer 0 the identity check; layer 1 the baseline assembly reproduces the stored card (Q, vert100); timing relative only (the BEV chat's lane loads the machine: record the load average and the Windows CPU share per repetition, a repetition above load 3 is marked, not dropped); the absolute value is repeated in the next idle window like E253.
- **Cost / kill**: 30 minutes of one CPU thread; kill as in the json.
- **Inputs**: ready (cached raw outputs of fast_recall, `p18_asm_check.py`, E253 driver pieces).
- **Referee**: not run for step 0 (a diagnostic); a code change that survives goes to an `ar-reviewer` pass (mode code) before it enters `training/`.

## 1b. Amendment after step 0 (2026-10-30, before any variant was measured)
- **Step 0 result** (`profile_step0.txt`, 200 frames, load average 4.2: shares are meaningful, absolute ms are not): of the 4.36 s cumulative, `relocate_forks` 21 %, `anchor_junctions_fast` 16 %, the path-union loop `place` 16 %, `simplify` 11 %, `prune_forks` 6 %. Two stages above 15 % (prediction of the card: hit), so the kill criterion does not fire.
- **Step 1a** (as in the card): change a stage, identical graphs. **Step 1b (added now)**: the two largest stages are post-steps that move vertices to junction-head peaks (`reloc_thr 0.4`, `junc_thr 0.5`); `closed_questions.md` says relocation / anchoring settings add nothing on top of fork pruning (E215, E218), but removing them outright was not tested. Variants on the cached outputs of fast_recall and champion, VAL 873: base, reloc off, anchoring off, both off; Q, J, split / merge F1, topo F1, vert100 and interleaved A/B assembly time (5 repetitions of 200 frames).
- **Rule for 1b**: a stage is dropped from the product (candidate, then the code route with `ar-reviewer`) only if Q loses at most 0.003, J at most 0.01 and vert100 changes by at most 0.3 on BOTH models, and the saving is at least 0.8 ms in relative timing; otherwise the stage stays. Timing is relative, provisional until an idle window.
- **Prediction for 1b**: relocation off saves 1.5 ms (tolerance 0.8) and costs J 0.01 (tolerance 0.01); anchoring off saves 1.0 ms and costs J 0.02.

## 3. Summary (2026-10-30, runs E258.p0 (profile), E258.a0 (step 1b), E258.b0 (step 2); VAL 873, cached raw outputs, CPU, load average 3.5 - 4.2 during step 1b; step 2 ran at load 5.4 - 9.1, above the rule of no timing claim above the physical cores (6): its timing is provisional)
- **Decision (by hand against the written rules, the metric is not a card metric): the profile found two stages above 15 % (prediction hit); no stage is dropped for the product as a whole; the junction anchoring is switched off for the j1 family as a candidate (registry entry `fast_recall_na`).**
- **Step 0, profile** (`profile_step0.txt`, 200 frames): share of the assembly: fork relocation 21 %, junction anchoring 16 %, path-union loop 16 %, simplify (spur / bubble) 11 %, fork pruning 6 %.
- **Step 1b, remove a stage** (rule: Q loss <= 0.003, J loss <= 0.01, vert100 change <= 0.3 on both models, saving >= 0.8 ms):

| model | variant | dQ | dJ | dvert100 | saving ms (relative, loaded) |
|---|---|---|---|---|---|
| fast_recall | relocation off | -0.0111 | -0.0540 | +1.41 | +2.3 |
| fast_recall | anchoring off | -0.0004 | -0.0042 | +0.04 | +2.7 |
| fast_recall | both off | -0.0141 | -0.0698 | +1.41 | +5.8 |
| champion | relocation off | -0.0110 | -0.0524 | +1.07 | +1.7 |
| champion | anchoring off | -0.0076 | -0.0356 | +0.06 | +2.3 |
| champion | both off | -0.0216 | -0.1007 | +1.09 | +4.9 |

  Relocation stays (it carries about 0.05 J on both models, the closed question E215 / E218 concerned its settings, not its removal). Anchoring is needed by the champion (the 288 / 144 chain) and nearly free for fast_recall.
- **Step 2, four seeds of the recall fine-tune (j1rc s0 - s3), anchoring off:** dQ -0.0004 / +0.0005 / -0.0010 / -0.0011; dJ -0.0042 / -0.0003 / -0.0068 / -0.0072; dvert100 +0.04 to +0.05; topo F1 +0.0046 / +0.0048 / +0.0040 / +0.0042; lanes found -0.0021 / -0.0022 / -0.0016 / -0.0023. The rule (all four seeds) holds; prediction (3 of 4) hit. The official card of the new entry (`evaluate.py fast_recall_na`, VAL 873): S 0.771, Q 0.739, J 0.443, topo F1 0.742, lanes found 0.734, R@1 far 0.726, vert100 15.02 against fast_recall 0.772 / 0.739 / 0.447 / 0.737 / 0.736 / 0.727 / 14.98.
- **Speed, honestly:** the saving is relative and was measured on a machine at load 3.5 - 4.2 (step 1b; step 2 up to 9.1) (base assembly 13.9 - 17.5 ms against 7.5 - 7.8 ms idle in E253). Anchoring is 18 - 26 % of the assembly in every pairing; applied to the idle 7.8 ms that is about 1.3 - 2 ms of the 18.5 ms chain (an estimate, not a measurement). The idle repeat belongs in the next quiet window (E253 driver with the new entry).
- **Weaknesses:** one base model family; the champion needs the stage, so a registry choice per variant (not a global switch) is needed; J is a smaller metric weight than Q but the D4 text quotes J, so the 0.004 - 0.007 loss of J must be stated; the timing carries no idle confirmation.
- **Not done:** an identical-graph optimisation of the stages themselves (step 1a): the profile suggests `relocate_forks` and `anchor_junctions_fast` as targets, both python loops over small arrays.
