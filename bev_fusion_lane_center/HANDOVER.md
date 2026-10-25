# HANDOVER - 2026-10-25 16:30

New chat: start in the repository folder (parent of autoresearch/ and this folder) or in this folder, say "Read HANDOVER.md and continue". Rewrite this file at every milestone (<= 3 KB).

## Where we are
- Loop 18 is CLOSED (10:55). Report: reports/weekly/Loop18_Fork_Pruning_Localisation_Report_EN.pdf (mailed twice). Results per node: experiment_execution/PLAN.md; every run: LEDGER.md; folders experiments/E209-E233. Headline: fork pruning closes the J gap (0.265 -> 0.409), chain 17.5 ms, joint E2E negative three times, recall weight buys lanes found not Q, TEST once (E230, S 0.748 vs 0.702).
- Decision D8 (a) is being carried out by the loop-18 lead chat ("BEV lane loop 18", owner of autoresearch/tools and evaluation/ until the commit "M0 + P5 + P1"): M0 golden numbers (`evaluate.py --golden write|check`, evaluation/golden.json: 6 registry names on VAL from the cache; the uncached champion entry on VAL250 waits for an idle GPU), P5 launcher hygiene (run.py: locked atomic ledger / schedule writes, snapshot through a temporary git index, no git identity -> refused, heartbeat / stalled, --timeout, GPU slots, id reuse refused, precise CUDA retry; selftest.py covers it), P1 experiment file (autoresearch/schemas/experiment.schema.json, tools/experiment.py check, `decide.py --experiment` reads the metric cards; five loop-18 experiments re-encoded in experiments/E215, E219, E232 give the recorded verdicts). Status: self-test passes, code review running, not yet committed.
- Data extension D7 (b) runs under the other chat ("BEV Fusion Spurmitte"): GaussianCaR features of the 255 new scenes, runs E234.tr0 / tr1 / val (2 GPU processes; ends about 17:30). Until that chat says "ledger free + GPU idle": no GPU job, no ledger writes, no change of the ledger format (its launchers rewrite the ledger when they end).
- The research system was simplified today (7 agents, 5 templates, prompts/research_steps.md; roadmap and the three reviews in ../autoresearch/ROADMAP.md).

## Next step
1. After the review: commit M0 + P5 + P1 (tools, schemas, evaluation, golden.json, experiment files), rewrite the golden entries once more so they carry the commit, update this file.
2. When E234.val is done and the ledger is free: `evaluate.py --golden write champion --subset val250 --fresh --tol <measured>` (two runs to measure the tolerance), `decide.py --experiment ... --write` for the five re-encoded experiments, `tools/maintenance/run_all_tests.sh`.
3. Then P2 (`ar tick`: reconcile + start ready schedule items within the GPU slots), P4 (hooks on the protected files), P3 (`ar metrics` with loop 18 as the baseline); order in DECISIONS D8.
4. Research cycle 19 by hand from `../autoresearch/prompts/research_steps.md` (P1 on the gaps of PROBLEM.md, P2 -> PLAN.md + schedule.csv); loop 19's GPU runs wait for the safety work.

## Decisions waiting for the user
In `experiment_execution/DECISIONS.md`: D3 folder rename, D4 fast deployment variant (champion decision with its TEST check), D5 purge of `_to_delete/`, D9 launcher identity check (done in P5, to be ticked).

## Pointers
Problem: PROBLEM.md; method: ../autoresearch/METHOD.md; closed questions: experiment_execution/closed_questions.md; literature: literature/OVERVIEW.md; old state: experiment_execution/history/; git: one repository (parent of this folder), code and text records, a tag per run (`lcr/<id>`), remote `origin` = github.com/Archimedes79/Auto-research-lmc (pushes only on request); golden numbers: evaluation/golden.json; experiment files: experiments/E<id>_*/experiment.json next to experiment.md.
