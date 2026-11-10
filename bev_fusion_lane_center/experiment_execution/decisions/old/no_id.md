# no_id

## Decided

Decided: 2026-10-25
Decision: git holds code AND all text records (no binaries, no data)
Consequence: `.gitignore` whitelist extended: experiments (text), experiment_execution, run records, model configs and metric cards, `data/splits.json`, literature summaries, report digests

## Decided

Decided: 2026-10-25
Decision: single-file records (ledger, decisions, closed questions, plan) stay complete; agents read experiment folders of the last two loops in full, older loops through loop / weekly summaries
Consequence: rule in `CLAUDE.md` and `autoresearch/METHOD.md`

## Decided

Decided: 2026-10-25
Decision: decisions come to Mirko through this file, not as file edits
Consequence: champion promotion, problem changes, tool / gate changes, permissions

## Decided

Decided: 2026-10-25
Decision: TEST is not decided separately: one TEST check belongs to a champion promotion, for the candidate and the current champion only
Consequence: the guard in the tools follows when M1 / M7 are built

## Decided

Decided: 2026-10-25
Decision: the outer loop (research cycle P1 / P2) is started by hand, step by step; the inner loop P3 runs autonomously
Consequence: done: `autoresearch/prompts/research_steps.md` (four steps started by hand)

## Decided

Decided: 2026-10-25
Decision: statistics: 2 seeds for screening, 3 seeds + scene-bootstrap interval before a champion proposal
Consequence: `decide` rule in M1 / M6

## Decided

Decided: 2026-10-25
Decision: no VAL split now (data is limited); see D1
Consequence: 

## Decided

Decided: 2026-10-25
Decision: one lead chat per loop owns the code; other chats work in `literature/` or `reports/` or read; `HANDOVER.md` names the owner
Consequence: 

## Decided

Decided: 2026-10-25
Decision: the safety work (M0-M2) and the cycle-19 research run in parallel; loop 19's GPU runs wait for the safety work
Consequence: 

## Decided

Decided: 2026-10-25
Decision: Mirko renamed the GitHub repository to `Archimedes79/Auto-research`; four workspace folders (autoresearch, bev_fusion_lane_center, bev_fusion, datasets/NuScenes; datasets/ and wsl/ not in git); raw nuScenes only on D: (`D:\Auto-research\NuScenes`), derived data of a problem in its project folder on C:
Consequence: `origin` set to the new URL (old URL redirects); raw data copied to D: and the CRN additions to `data/nuscenes_additions/` of this project; WSL keeps link "views" at the old paths (`datasets/NuScenes/tools/link_views.sh`); the folder rename D3 is still open (after it, re-run `../../datasets/NuScenes/tools/link_views.sh`)
