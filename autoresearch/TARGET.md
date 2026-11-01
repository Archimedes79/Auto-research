# autoresearch: the target design (D32, 2026-11-01)

The goal of the restructuring that starts here: adopted by Mirko on 2026-11-01, **not built yet**. Today's system is drawn in `architecture.svg`; this page says where it goes and in which steps. It replaces the target packages of `ROADMAP.md` once step 1 below is done.

## Four layers
| layer | what is in it | what it does |
|---|---|---|
| Person | Mirko, `decisions.md` | gives the prompt, approves the problem, promotes a version to champion, allows the one TEST check, grants permissions |
| Agents | four roles: **planner** (modes: problem draft, branches, run once per hypothesis lens, detail plan, tracking), **researcher** (scouts in parallel, summaries), **coder**, **critic** (fresh eyes on cards, code and numbers: the former referee and reviewer) | judgment only; an agent reads files and writes files, nothing else passes |
| Files | the problem folder (below) | the only contract between agents and code |
| Code | `core/` (launcher, gates, decision rule), `suites/train/` (supervision with kill rules), `suites/eval/` (validation, ms per frame, `package`), `report/` (board, numbers check) | rules, runs and measurements; no judgment |

Transition points between the layers, each a check that can refuse: the **write gate** (protected files, complete cards), the **start gate** (a card, a free GPU slot, a tagged snapshot; the TEST split only with a logged row), the **decision rule** (KEEP / CUT / TIE / INCONCLUSIVE from the card's clauses), **promotion** (Mirko, with one TEST check) and the **cleanup** at the end of a loop (scratch tagged, then moved away).

## Two loops
- **Outer loop**, one round per prompt: problem (drafted by the planner, approved by Mirko) -> research -> branches (experiment cards) -> the inner loop for each card -> tracking (updates the set of n best versions by the written rule) -> promotion with one TEST check -> cleanup.
- **Inner loop**, one per card: detail plan (names the boxes of the system drawing it changes) -> code (reviewed) -> train -> evaluate on VAL -> decision rule -> verdict. A variant experiment runs it without a chat.

## One unit of data: the experiment folder
```
<problem>/
  problem.md          goal, metrics, protocol, known gaps (approved by Mirko)
  problem.json        code folders, train command, evaluate command (writes a metric card), holdout split, metric directions, GPU budget
  architecture.json   the system drawing; its main path counts the rule steps between input and output
  research/           paper notes, group summaries with their digests
  experiments/<id>/   card.md (with its machine block: arms, seeds, metric, band, rule, parent version, boxes changed)
                      runs/<arm>.<seed>/run.json + card.json (metric card)   verdict.json   the summary, back in card.md
  versions/<name>.json  the n best: tagged commit, config, weights, metric card, status (best / champion / retired)
  decisions.md  handover.md  scratch/
```
Written by hand (agents or Mirko): problem, cards, versions, decisions, handover. **Derived by code and never edited:** the ledger, the board, the schedule ("what can start" = cards without a verdict whose dependencies have one), the plan status, the per-loop table. A problem plugs in with `problem.json` (about 6 keys instead of today's 25) and two commands; everything else is a fixed folder name.

## What the design must reach
Layers visible (a folder per layer, `archdraw.py check` keeps drawing and folders together) - information flows only through experiment folders and `versions/` - rules in code at every transition point - lean (4 roles, 1 core, 2 suites + report, 5 kinds of hand-written file, method on one page, one page per role) - TEST once at promotion - every state rebuilt from files after a crash - the E2E gaps of each version on the board.

## Migration, in this order
Each step ends with the selftests green, `architecture.json` redrawn to the new state and checked, and one real run through the changed path. Running chains keep working until a step replaces their path.

| step | change | needs |
|---|---|---|
| 1 | roles to four: planner modes, critic = referee + reviewer (agent texts, `install_agents.py`, METHOD.md to one page) | this decision |
| 2 | `eval package`: run a version on frames, measured ms per frame, the output format of the problem as a small adapter function; replaces `bev_fusion/evaluation/bev_prototype.py` and `bev_fusion_lane_center/tools/prototype/lane_prototype.py` | `evalsuite` free of other chats' uncommitted work |
| 3 | the experiment folder as the unit: ledger, schedule and board derived from it; `schedule.csv`, `tick.py` and the hand-kept views go | this decision |
| 4 | `versions/` replaces lane `models/registry.json` and bev_fusion `records/champion.json` | a decision: protected champion entries |
| 5 | layer folders in `autoresearch/` (`core/`, `suites/`, `report/`, `contracts/`) and `problem.json`; bev_fusion first, the lane project after one full loop on the new paths | a decision: `project.json` is protected; no chain may be running |

Not part of it: new suites, a service or daemon, templates beyond the card and the version entry.
