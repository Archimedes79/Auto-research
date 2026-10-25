# Loop 16: refined loop description (paste text)

Prepared 2026-10-23. Nothing is started by this file. To run it later: type `/loop` (no interval, so the loop paces itself) and paste the block below; fill the three parameters in the first line. The block is delivered again at every wake-up, so everything long lives in files that it points to (plan, spec, concept, experiment table, state file).

Files the block relies on (all exist now):

| file | role |
|---|---|
| `reports/Loop16_Triage_Plan_EN.md` | search design (L8 array, forked L4 arrays, halving), gate, scoring, refinement rules, loop protocol, tracking tables, technique register K01 - K133 |
| `out/loop2/e123_ladder/TRIAGE_STATE.md` | live state and tracking tables (the loop reads it first at every wake-up) |
| `reports/Hypothesis_Tree_Spec_and_Tests_EN.md`, `scripts/p16_ref.py`, `scripts/test_p16_spec.py` | the design under test, 18 spec tests passing on a numpy reference |
| `reports/Concept_Hypothesis_Tree_Lane_Graph_EN.md` | concept, winning concepts from the literature, mixes M1 - M12, baseline arms B0 - B3 |
| `reports/Experiment_Table_EN.md` (+ `out/loop2/EXPERIMENT_TABLE.csv / .json`) | all 80 earlier experiment rows with building blocks, technique ids, links and the sparsification stage; the sparsification map |
| `report_final/text/lit_p16_tree.md`, `lit_extra15.md`, `lit_p16_links_a.md`, `lit_p16_links_b.md` | literature and verified links |

## The block

```
STOP_AT: <HH:MM local, e.g. 04:00>     SHUTDOWN_AT_STOP: <yes | no>     MAIL: no   (mail only if this line says so)

MISSION
Find, test and document a fully end-to-end differentiable architecture for lane-level centre-line detection, SPARSIFICATION and map assignment, whose output (a sparse lane graph with a navigation attribute per lane) can be handed to a planner and trained through. E2E means all modules for the sub-functions can be connected into one deep network and trained / refined end to end, so every module needs a differentiable form; only differentiable solutions enter (the gate of the plan decides). This loop concentrates on SPARSIFICATION: the fewest vertices and edges that still give a correct graph ("better short lane graphs than false lane graphs"). Starting point (measured): E94 / E120 has 22 vertices per 100 m against 14.3 for a Douglas-Peucker oracle (the Imai-Iri floor is about 13), split / merge F1 0.36 / 0.46 against 0.69 / 0.65 for the rule system, seeds as the limit. Map assignment, temporal fusion and the planner interface are tested on the winning design at the end (rung 3).

HOW THE LOOP RUNS (state-driven and systematic; no free wandering)
Everything is planned in files: reports/Loop16_Triage_Plan_EN.md (search design, gate, scoring, refinement, tracking), reports/Hypothesis_Tree_Spec_and_Tests_EN.md (design under test, spec tests), reports/Concept_Hypothesis_Tree_Lane_Graph_EN.md (concept, winning concepts, mixes, baseline arms), reports/Experiment_Table_EN.md (earlier experiments with basis, sparsification map), report_final/text/lit_p16_tree.md and lit_extra15.md (literature), out/loop2/LOOP15_INDEX.md and loop2/STATE.md (earlier results).
At every wake-up: (1) read out/loop2/e123_ladder/TRIAGE_STATE.md first (phase, rung, designs alive, jobs, next action) and act on it, never re-plan from scratch; (2) check running jobs (logs, pids), parse finished runs, update the tracking tables; (3) if a rung is complete: cut the worst 50 % by the rules of plan section 5, write the rung summary, prepare the next rung; (4) launch the next jobs (at most 2 GPU jobs, wrapper scripts with Start-Process wsl as in loop 15, GPU memory below 11 GB); (5) write the next action into the state file; (6) ScheduleWakeup in 20 - 25 min with this same prompt. When nobody answers, decide by the plan and record the decision and its reason.

PHASES
P0 harness: write p16_builder.py (switches A - G, spec geometry), p16_gate.py, p16_train.py, p16_diag.py, p16_triage.py, p16_report.py; add the tests T26 - T29 of the concept to the numpy reference; all spec tests pass. While coding, run the control arm E122 (E94 + E120, seeds 0 and 1, fidelity curve).
P1 gate: admission gate G0 - G4 for D1 - D8 (no argmax or matching on the training path; gradient reaches every parameter group, also from the planner loss). A design that fails is fixed or replaced, never run.
P2 ladder = successive halving. Rung 0: the 8 designs of the L8 array, trial + 1 refinement. Rung 1: 4 survivors, forked L4 over the sparsification factors SPF1 - SPF3 (step-length rule, validity / keep gate with sparsity penalty, reach of the ray ladder), plus the hybrid H challenge. Rung 2: 2 survivors, forked L4 over HF1 - HF3 (stage B refine, stage C rescore, b = 3 hypotheses). Rung 3: the winner with time recursion, joint end-to-end fine-tuning, route head and consumer (two seeds), map-fault protocol, TEST once. Cut the worst 50 % after every rung. Baseline arms B0 - B3 run beside the ladder (control, geometric link, nodes + edges, single-score verifier). Select on VAL only, fixed seeds; a difference below the noise band delta is a tie, not a result.
P3 report.

TRACKING (never skipped)
- Before a run starts write its ledger row (id E123.<design>.r<rung>.<step>, hypothesis, change, parent, fidelity, seed, basis K ids); when it ends fill metrics, S, dS, flags, decision and reason. Failed, killed and cut runs stay in the table with the reason: they are results for the report.
- After every rung: cut log, coverage ledger, pooled effects, frontier points (F1@1 against vertices per 100 m, FL@R85, F1@Sp17), rung summary with three figures.
- Every finished run is appended to the experiment table (scripts/e2e_catalogue.py and scripts/e2e_basis.py, ids from E124, tag E / P / C / D, SP stage, BB and K ids) and build_experiment_table.py is re-run, so the table always says what was tried on which basis. A technique is entered in the register (K id, source, link, checked or from memory) before it is used.
- Coverage: no factor level may end the loop with fewer than 2 finished runs unless the reason is written in the coverage ledger.

BUILDING BLOCKS AND LITERATURE
Deep learning is a catalogue of building blocks: convnets, GRU or state-space memory for temporal fusion, blocks behind each other for iterations, self-attention (every list element with every other), cross-attention (one list pulls from another), pointers, gates that fuse the input with the pulled data, deformable and ray sampling, survival and hypothesis blocks. The catalogue is the BB table of the experiment table (30 blocks); extend it with source and K id when you find or build a new block, and enumerate the additions shortly in the report. Before the ladder starts spend at most 45 minutes on a delta search (2025 - 2026: lane graph, sparsification, hypotheses, map assignment), abstract pages only, no downloads, mark every entry checked or from memory, never invent numbers.

DESIGN GUIDE
Sparsification is the target: step length emerging from the verification, learned validity and keep gate with a sparsity penalty (hard concrete), reach of the ray ladder, n best hypotheses, tree to DAG. Run the baselines from the literature (TopoLogic-style geometric links, SAM-Road-style nodes + edges, LoI single-score verification), because they may simply win. Report precision-first numbers (FL@R85) next to F1. Rough localisation (translation and rotation) stays inside the route-head chain (E103e, E104b) and is varied only at rung 3 together with the ADASIS-like simulated SD map and its error protocol (E77, E87): assigning the navigation information after perception is expected to remove many map problems; test that, do not assume it. Build on previous results: start from the sparsification map of the experiment table.

EVALUATE DEEPLY
After every rung run the diagnostics (drift along the chain, along / across error, lane recall by length, duplicates, split / merge landmark precision / recall at 1 / 2 / 4 m, fork-angle resolution, vertices per 100 m, calibration of the validity), name each weakness, state one hypothesis per weakness and the refinement it implies (diagnosis table of the plan). Understand why a design lost before it is cut and write the reason.

CONSTRAINTS (standing)
Only fully differentiable designs. No downloads or installs without asking (reading web pages is fine). Never delete user data or earlier results; archive instead. At most 2 concurrent GPU jobs. Two seeds for any route-head comparison (its seed noise is up to 0.03 route success). TEST (21 scenes) is used once, for the winner and the control. The setting uses exact poses and a simulated SD map; say so in the report. No mail unless MAIL says so. At STOP_AT: write the report, update the memory file (facts only), STATE.md and the state file; if SHUTDOWN_AT_STOP is yes, first check that no other Claude session or agent is running (ListAgents), then shut down; end the loop (ScheduleWakeup stop). The whole ladder needs about two nights: at STOP_AT leave the state so that the next start with this same prompt continues where it stopped.

REPORT (English, written along; final version at STOP_AT minus 45 minutes or when the ladder is finished)
Orient yourself on high-quality papers. Skeleton in plan section 8: task; building blocks; literature and register; earlier results and the sparsification map; search design and coverage; results by rung; effects and the frontier; the best design (architecture drawing without red blocks, spec-test and gate results, VAL and TEST against the control, route and planner numbers with two seeds, pictures from the output, qualitative and quantitative conclusions); what did not work and why; constraints and limits in very short sentences; next steps. Concentrate on the most impactful experiments and the best methods so far. For every experiment: architecture, hypothesis, basis ids, result, pictures, conclusion. Send the key images to the user in the chat (SendUserFile). Put the report files into reports/ with a README line.
```

## What changed against your description

| part of your description | kept | refined or added |
|---|---|---|
| E2E, only differentiable solutions | yes | made testable: an admission gate with a static scan, gradient-reach tests (also from the planner loss) and spec tests; designs that fail are not run |
| building blocks, bigger list from literature, enumerate shortly | yes | the 30-block catalogue (BB table) exists and is extended with source and K id; delta literature search limited to 45 minutes before the ladder |
| latest literature, build on previous results | yes | register K01 - K133 with links; the experiment table carries building blocks, technique ids and the sparsification stage of every earlier experiment; sparsification map as starting point |
| localisation, pixel map first then sparse graph, short rather than false graphs, connect to the map, temporal fusion, ADASIS | yes | the chain is fixed (dense head, sparsifier under test, route head, consumer); the focus is the sparsifier; localisation, map faults and time are tested at rung 3; precision-first metrics (FL@R85) added |
| systematic development and fine tuning | yes | successive halving over an L8 array, forked L4 arrays for sparsification and hypothesis factors, control arm, noise band, VAL only, TEST once; two nights |
| evaluate deeply, hypotheses for improvements | yes | automatic diagnostics after every rung, a diagnosis table that maps each weakness to one refinement, reasons recorded for every cut |
| propagate a graph with navigation information to a planner | yes | rung 3: route head E104b and consumer E117 with the winning graph, two seeds |
| report in English with architecture, hypothesis, result, pictures, conclusion, limits | yes | the report is written along from the tracking tables; every claim names its basis ids; skeleton in plan section 8 |
| stop at a time and shut down | parameterised | STOP_AT, SHUTDOWN_AT_STOP and MAIL are explicit; shutdown only after a check for other sessions |
| (new) tracking and coverage | - | ledger per run, cut log, coverage ledger, pooled effects, frontier points, rung summaries; every finished run goes into the experiment table |
