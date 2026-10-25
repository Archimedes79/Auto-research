# Next loop (refined prompt, short form) - NOT started

Paste as: `/loop <text below>` (self-paced, 20-25 min wake-ups)

---

STOP_AT: Sunday 11:20 | SHUTDOWN: yes, only if no other chat runs (ListAgents) | MAIL: yes, report PDF to mirko.meuter@aptiv.com via web.de in Chrome (as loops 10 / 13) | TEST (21 scenes): OK, once

## Goal
- Best fully differentiable chain: perception -> centre lines -> sparse lane graph -> temporal fusion -> map assignment -> navigation attribute -> planner.
- E2E = all modules in one network, modular training first, then longer joint refinement. Only differentiable parts on the training path (no argmax, no matching, no rule step); gradient reaches every module.
- Quality AND time: every experiment reports ms per frame (idle GPU, batch 1, cached foundation features).
- Better short lane graphs than false lane graphs.

## Start from (read first, never re-plan)
- State: `out/loop2/e123_ladder/TRIAGE_STATE.md` (loops 16 / 16b), `out/loop2/e17/LOOP17_STATE.md` (loop 17), `loop2/STATE.md`, memory `bev-lane-camera-radar-state.md`.
- Literature already read: `report_final/text/lit_p16_*.md`, `lit_extra15.md`, `lit17/` (103 cards), `reports/Literature_Update_Loop17_EN.md`.
- Loop 16 (tree search D1-D8): champion D7 (+ junction head, time recursion). VAL S 0.746 vs control 0.724; TEST S 0.728 vs 0.702; 13.5 vs 21.8 vertices / 100 m. Weak: J 0.27 vs 0.40, lanes found 0.70 vs 0.80.
- Loop 16b: route head along the edges 0.889 (vertex head 0.798, control 0.885); chain 31 ms (single frame j1 192/96: 25 ms) vs control 628 ms; shallow forks = annotation limit.
- Loop 17 (other chat, taken over with the user's OK): use all results up to the E209 crash (night / blur / cameras off; hold, gate, map builder, consumer; numbers in its state file). E209 (unseen failure types) crashed at 21:00 (CUDA unknown error, no checkpoint); `e142.done` is FALSE, delete it.
- Open: TEST run of the route head along the edges + fast configs (now allowed); j1 192/96 as deployment setting.

## Every wake-up
- Read the state files; check jobs (max 2 GPU jobs, GPU < 11 GB, wrapper scripts, checkpoint every 400-500 its); parse results; write the next action; schedule the next wake-up.
- Ledger row per run, catalogue row (E-id), K-id per new technique, 2 route-head seeds per comparison (seed spread up to 0.09).
- First wake-up: gap check = what was done (ledger, experiment table T9), what the literature says is missing, what to rerun.
- Never kill another chat's jobs unless I said so; check ListAgents first.

## Work packages (in order)
1. **Finish loop 17**
   - rerun E209 with checkpoints, evals, report text, one fresh numbers reviewer, PDF to `reports/`.
2. **Building blocks (short list, 2025-26)**
   - convnets, GRU / SSM memory, iterative blocks, self-attention, cross-attention (pulls from another list, fuses with the input), pointer / deformable sampling, gated fusion, survival / hypothesis blocks.
   - abstract pages only, no downloads, mark "checked" / "from memory".
3. **Tree search round 2 (3-5 candidates, each: hypothesis, gate, 1 trial + 1 refinement, ms)**
   - joint E2E fine-tune after modular training, longer (4-8 u): builder + gate + route head + consumer, with failure frames.
   - grid head (dense lane pixel map) + sparsification trained E2E: one loss from graph / route back into the dense head, against the modular version.
   - rough localisation (translation + yaw): differentiable registration of the sparse graph to the SD map; target: map benefit at 1-2 m error.
   - map trust / abstain output (dustbin column) for wrong or missing maps; extend lanes with map information.
   - second-successor head for forks, judged with a geometry-following fork metric.
   - speed: compiled assembly, CPU / GPU overlap.
4. **Deep evaluation**
   - per weakness one hypothesis, one test; honest negatives stay in the report.
5. **Report (English, along the way)**
   - skeleton: task, building blocks, literature + register, earlier results, search design + coverage, results by rung, effects + frontier, best design (architecture drawing, no red blocks), VAL / TEST vs control, route / planner numbers with 2 seeds, ms table, what failed and why, limits (very short), next steps.
   - per experiment: architecture, hypothesis, basis ids, result, pictures, conclusion.
   - good pictures; key images to me with SendUserFile; PDF into `reports/` + README line.

## At STOP_AT
- finish report, memory (facts only), STATE files; mail the PDF; shut down only if ListAgents shows no other chat running; stop the loop.
