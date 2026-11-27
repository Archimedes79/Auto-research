# Autoresearch: the restructure plan (2026-11-26)

**Status (2026-11-27, D154 / D155):** carried out in the branch `v2`. S0 structure; S1 the core `autoresearch/research/` with the toy as its chain
test, two reviews (architecture, code) and their fixes; S2 the tag `v1` (main plus its uncommitted records); S3 data and runs on D:
(`D:\Auto-research\data`, `D:\Auto-research\runs`), the code reads them through the machine file; S4 BEV and S5 lane: the champions' code
moved with `git mv`, their old folders, tools and records left `main` (they are in `v1`). Section 13 says what was measured and deleted. Where
the sections below plan a worktree per experiment and an adopt that applies a diff, the built system differs: an experiment's code is
committed default-off before its start, `start` refuses uncommitted code, and `adopt` makes the arm's settings the defaults (the reviews
showed a diff of a shared checkout picks up other chats' edits).

**The owner's frame (2026-11-26):** no backward compatibility; keep the code of the most successful experiments and build on it, no fresh start; the old decisions and most old experiments are not needed; git is the archive, no old folders are kept; code and data apart (data may go to D:, code does not); run outputs go to `runs/`; a champion must be reproducible through git, not every intermediate result; predictions are stored, and the code in git regenerates a report, its pictures included, from them and the raw data; a decision belongs to its problem; "ground truth", not "truth". Where this page differs from `TARGET_2026-11-26.md` and `P0_PLAN_2026-11-26.md`, this page counts; their sections on other machines and on data shards over ranks stay.

## 1. What the reviews found
Two independent reviews on 2026-11-26: the architecture of the proposed system, and the code of the toy problem (reproduced in throwaway copies). Code findings marked * were read again in the code afterwards.

| architecture finding | answered in |
|---|---|
| A1 the rule is not frozen: its hash covers neither the arms' commands nor their settings (`experiment.py`, `rule_sha`) | section 4: the judge reads the snapshot |
| A2 TEST is guarded at the start command, not where data are read | 4: holdout-aware reader |
| A3 a control that did not train "loses" and the variant is kept | 4: validity floor, INVALID |
| A4 the paired comparison sits in the evaluation of a run | 4: it moves into the judge |
| A5 a card cannot hold lane work (E395: nine arms, chained runs, its own verdict script) | 4: step graph, stored references, frozen `judge.py` |
| A6 records too large (BEV cards, median about 840 KB), state without a single writer, no data-set contract, champions defined in several places and differently | 4, 6, 8 |

| code finding (toy) | severity | fix, in the core (step S1) |
|---|---|---|
| C1* stored predictions are reused although `net/` or the runner changed (identity is config and checkpoint; the evaluator hash omits both) | blocker | identities cover all code that shapes a prediction; a record never reuses what a call by hand wrote |
| C2* the ground-truth store goes stale or "heals" a damaged shard (no data hash in the signature; a present file is kept and recorded again) | blocker | the signature includes the data manifest; a differing file is refused |
| C3* a `--no-card` diagnostic can overwrite a record card, against hard rule 3 (`AR_COMMIT` is set for it too) | blocker | only the worker of a registered run writes records, into its own folder; "diagnostic" is said, not guessed |
| C4* `.gitignore` `runs/` also hides `records/runs/*/run.json`: the records behind a verdict would never be committed | blocker | no ignore rule that can match a record; a test asks git about every record path |
| C5 resume and warm start silently accept changed ground truth, data or loss weight; an old `.done` survives a failed repeat | important | resume compares data and ground truth; the worker moves old state away |
| C6* TEST is open in the template (every split offered); "dirty" answers clean when nothing could be checked | important | holdout only through the gate; unknown is not clean |
| C7 the selftest passed with planted bugs (swapped confusion arguments, transposed ground truth, reversed paired difference, raw instead of averaged weights); the adapter re-implements the forward pass, so an edit of `forward` is not evaluated; the report's steps per second are about 10 % low | important | asymmetric fixtures, asserts on the sign of the difference; call the model; cost from run totals |
| C8 the record card is written before comparison and pictures; an unknown path resolves by its basename; the launcher's check builds data | minor | with S1 |

The code review did not cover the report build, the verdict tool on E1, the drawing, or concurrent writers beyond reading. Generic glue that no problem should copy (scoring step, arm wrapper, launcher script, cost table, card and provenance, data and ground-truth stores) goes into the core.

## 2. Code and data
| what | where | in git |
|---|---|---|
| code and text: `autoresearch/`, `GtGenerator/`, `datasets/`, `problems/`, `champions/` | `C:\Auto-research` | yes |
| raw data, ground truth, caches | `D:\Auto-research\data\` | no |
| run outputs: checkpoints, logs, predictions, full cards, built reports | `D:\Auto-research\runs\` | no |
| the packed old records, checkpoints and run tags | `D:\Auto-research\archive\` | no; listed in a catalog in git |
| machine state: the two roots above, GPU slots, waiting line, holds | `~/.research/` | no |

The checkout holds no data folder. Code reaches data and run outputs only through the two roots of the machine file (`~/.research/machine.json`); a problem never names a path. On this PC the roots are on D:, on a rented machine they are wherever its disks are. The code part of today's `data/NuScenes/` (README, manifest, download and check tools) becomes `datasets/nuscenes/`.

**Speed.** WSL reads D: through the Windows file bridge. Caches that a training reads every step (the 161 GB lane folder `~/exp_gcar_b123` lies inside the WSL disk image today) may become slower there. Before they move: one training epoch measured both ways. If it is too slow, a Linux disk file on D: mounted into WSL keeps the speed (a system change: the owner's OK).

## 3. Architecture
| layer | folder | owns | never |
|---|---|---|---|
| core | `autoresearch/` | experiments, runs, records, gates, comparison, verdicts; the training engine; the evaluation engine, runner, evaluation step and result writer, cost; reports; the data roots | names a problem or a data set |
| data-set services | `GtGenerator/`, `datasets/nuscenes/` | what a data set is: splits by scene name, hashes of raw data, ground truth and caches, which scenes are the holdout | knows a network |
| problems | `problems/<name>/` | what the data mean and what "right" means: targets, model, loss, metric, views, score | launches, records, judges, writes provenance |
| champions | `champions/<name>/` | the facts of the best network and the comparison networks | holds code: the code lives once, in the problem, and exactly in the tag |
| data, runs (not in git) | `D:\Auto-research\{data,runs}\` | the bytes, named by hash or signature | holds a fact that no record names |

**A problem writes only this:** `problem.json` (name, tag prefix, the score with its guards and the floor a valid control must reach, splits and holdout, protected files, environment, the steps), `training/train.py` keeping the run-folder contract (`trainkit` recommended, not required), `evaluation/adapter.py` (source, subject, metrics as additive counts) and `evaluation/views.py` (one frame drawn from raw data, ground truth and a prediction). Everything the judge trusts is core code. In the toy about 340 of 890 non-test lines were glue every problem would write again.

```
 owner: orders, decisions
   |
 lead / conductor / reviewer (agents)                 write experiment.md and code; nothing else
   |
 research  new check start status stop resume compare judge adopt report index      (autoresearch/research)
   |  start: gates -> ONE snapshot commit and tag per experiment -> run folders
   v
 worker, per run, from the snapshot: the steps of the arm (prepare -> train -> evaluate -> ...)
   |      slots, memory, kill rules, heartbeat, resume          training: trainkit; evaluate: core step + adapter
   v
 runs/<problem>/<run>/ (D:)                  experiments/<E>/runs/<run>/  config.json  run.json  result.json   (git)
   |
 judge: card and problem.json of the SNAPSHOT -> control valid? -> paired comparison from stored counts -> verdict.json
   |
 adopt (the owner's word): the winner's code from the tag onto main -> anchor test -> champions/<problem>/
```
`index` generates `ledger.csv` from the experiment folders; `report` builds from records, champion facts, stored predictions and raw data.

## 4. Interaction
| verb | who | reads | writes |
|---|---|---|---|
| `new <problem> "<question>"` | lead | template; the next id (under a lock) | experiment folder with `experiment.md` |
| `check E` | conductor, reviewer | card, `problem.json`, data manifest | nothing |
| `start E [--seed n]` | conductor | the same, the TEST log, holds | snapshot commit and tag; per run `config.json` |
| (worker) | the machine | the snapshot | `runs/`; `run.json`, `result.json` |
| `status [E]`, `stop`, `resume` | anyone / conductor / only with the owner's words | run files, heartbeats | stop request, hold |
| `compare A B` | anyone | stored counts | a diagnostic table |
| `judge E` | conductor | the snapshot's card and rule, results and counts of all arms | `verdict.json`, the ledger; one commit of the records |
| `adopt E` | conductor, on the owner's word | the tag, the verdict | the code onto main, `champions/<problem>/`; one commit |
| `report <spec>` | anyone | records, champion facts, predictions, raw data | the report into `runs/` |
| `index` | anyone | all experiment folders | `ledger.csv` (generated, never edited) |

**One fact, one file, one writer.**
| fact | file | writer |
|---|---|---|
| question, arms, rule, conclusion | `experiment.md` (the card block is what the snapshot froze) | the agent |
| the exact config of a run | `config.json` | `start` |
| what happened to a run | `run.json` | the worker |
| its numbers, evaluator hash, ground-truth signature, data hashes, compact per-scene counts | `result.json` | the evaluation step |
| the verdict, the interval, why | `verdict.json` | the judge |
| the champion and the comparison networks | `champions/<problem>/` | `adopt`, on the owner's word (protected) |
| permissions | the problem's `DECISIONS.md` and `TEST_LOG.md`; the root `DECISIONS.md` for what concerns several problems | the owner decides, agents record |

**Rules of the flow** (each answers a finding of section 1 or of P0):
- **The rule is what was committed.** The judge reads card and `problem.json` from the experiment's snapshot; a later edit of arms, guards or settings does not count.
- **A control must be valid.** Below its floor, or with seeds further apart than the card allows, the verdict is INVALID, not KEEP.
- **The comparison belongs to the judge:** runs are independent, stored references can be reused, the interval resamples scenes and seeds.
- **A complex rule may be a script** (`judge.py` in the experiment folder, part of the snapshot, so written before the run). A verdict by hand stays possible and is marked.
- **Steps form a small graph:** a step names what it needs (an earlier step, a stored run) and its resources (`{gpus: n}`).
- **The holdout is guarded where data are read:** the core reader and the evaluation step open holdout scenes only when the TEST gate says yes for this run.
- **Identity covers what shapes a result:** code, config, weights, ground truth, data. Anything stored is reused only under the same identity; a stored file that differs is refused.
- **Only a registered run writes records,** into its own folder, under one id spelling. Whether a call is a diagnostic is said explicitly, never read from the environment.
- **Records stay small:** `result.json` holds score, guards, read-outs and per-scene counts; full cards stay in `runs/`.

## 5. Git
| what | where |
|---|---|
| the champions' code (in the problem folders), `champions/`, the records of finished experiments | `main` |
| exactly what ran: code, card, the experiment's scripts | tag `<problem>/E<id>`: one snapshot commit per experiment through a temporary index, never merged |
| everything before the switch | tag `archive/pre-restructure`: the tracked files plus the records and scripts that are not in git today; pushed on the owner's word |
| bytes | D:, never in git |

1. `new` and the card: nothing in git yet.
2. A code change is made in a git worktree of the experiment (`temp/wt/E###`), so `main` and other chats stay untouched. A change of settings only needs none.
3. `start`: the snapshot commit and tag from that tree plus the experiment folder; the runs execute the tag, nothing else.
4. `judge`: one commit on `main`, records only (experiment folder, ledger).
5. KEEP and the owner's yes: `adopt` applies the experiment's code change to today's `main` and commits it only when the anchor test reproduces the experiment's numbers there. CUT: nothing is merged; the tag keeps the code.
6. Building on an experiment that was not adopted: start the worktree from its tag.
7. Push only when asked: fetch, check, tags one by one.

## 6. Folder structure
```
C:\Auto-research\                 code and text only
  CLAUDE.md  README.md  .gitignore  .claude/
  DECISIONS.md                    what concerns more than one problem (push, gates, this switch); from S6
  autoresearch/
    README.md  METHOD.md          how to use it, the rules
    research/                     cli, spec, launch, worker, judge, adopt, index, guard, gitio, clean
    trainkit/                     training engine (later: ranks and shards)
    evaluation/                   engine, generic runner and evaluation step, statistics, cost
    data/                         the roots, manifests, signature stores, the holdout-aware reader
    report/                       the report tool (from reporter/): tables, pictures through the problem's views, html
    agents/  templates/  schemas/  tests/
  GtGenerator/                    makes the ground truth, as today
  datasets/nuscenes/              today data/NuScenes/: README, manifest, download and check tools
  problems/
    toy/                          reference problem and chain test (here since S0)
    lane_center/                  Spurmitte, today bev_fusion_lane_center/ (from S5)
    bev_fusion/                   BEV Fusion, today bev_fusion/ (from S4)
  champions/                      protected, no code
    lane_center/  bev_fusion/
  temp/  wsl/                     not in git

D:\Auto-research\                 bytes only, not in git
  data\
    nuscenes\                     raw (today NuScenes\, NuScenesAdditionalGt\)
    groundtruth\<signature>\      today C:\Auto-research\data\autogt\
    cache\<problem>\<signature>\  features and caches (today NuScenes_cache\, the problems' data\ and runs\ caches, ~/exp_gcar_b123)
  runs\<problem>\<run>\           checkpoints, logs, predictions, full cards, built reports
  archive\                        packed old records, checkpoints, the bundle of the old run tags
```
```
champions/<name>/
  champions.json      who is champion, who is a comparison network, and the decision for each
  <network>/
    network.json      tag and commit, config, weights in runs\ with sha256, the command; numbers per metric with split
                      and denominator; evaluator hash, ground-truth signature; timing (ms per frame, parameters, machine)
    counts.json       per-scene counts: paired comparisons and report tables without running the network
    architecture.svg  the drawing (text, made by archdraw)
```
```
problems/<name>/
  PROBLEM.md  DESIGN.md  DECISIONS.md  HANDOVER.md  CLAUDE.md  TEST_LOG.md
  problem.json        score and guards, splits and holdout, steps, environment, protected files
  ledger.csv          generated, never edited
  dataset/            manifest, splits by scene name (text)
  gt/                 targets the problem makes itself (nuScenes ground truth comes from GtGenerator)
  net/                prep/  model/  loss/
  training/           train.py, configs/, presets/, architecture.json
  evaluation/         metrics, adapter.py, views.py
  experiments/E###_slug/          new experiments only
      experiment.md               card, idea, conclusion
      judge.py                    only for a rule the standard judge cannot say
      verdict.json
      runs/<run>/                 config.json  run.json  result.json
      report/                     report spec and the author's text
      *.py                        code only this experiment needs
  reports/<name>/     report spec and text of a report that is not one experiment's (e.g. the state of the problem)
  tests/              including the anchor test: the code on main reproduces the champion's numbers
  literature/         as today
  temp/               ignored
```
Gone compared with today: `experiment_execution/`, the old experiment folders, `records/`, `results/`, the problems' `runs/` and `data/` (to D:), `winner/` (code in the problem, facts in `champions/`), `runner/`, `models/registry.json`, `tools/` and report builders per problem, `schedule.csv`, the hand-written ledger, `reporter/` (into `autoresearch/report/`), the checkout's `data/`.

## 7. Reports and reproducing a champion
**A report is regenerated, never kept.** Its spec and the author's text are in git (`report.json`, the parts). The code is in git: the core report tool (tables, layout, cost, number check) and the problem's `views.py` (one frame from raw data, ground truth and a prediction). The inputs are the records and champion facts (git), the stored predictions (`runs\`), raw data and ground truth (`data\`). Building it runs no network; `--predict` runs the weights again when predictions are gone, and ground truth of an older signature is rebuilt from the `GtGenerator` tag of that version. Test: built twice, the same numbers and pictures; the report names the signatures it used.

**A champion is reproducible at three levels.**
| level | from | result | when |
|---|---|---|---|
| report | stored predictions, raw data | numbers and pictures, exactly | any time |
| evaluation | weights (sha256 in `champions/`) and the code on `main` | the numbers, bit for bit | the anchor test, at every change of problem code |
| training | code, config, data manifest and ground-truth signature in git | the numbers within the seed band | at adoption and at the switch, on GPU time the owner gives |

Intermediate results are not kept, except the weights the champion's chain takes as inputs (lane: the lane head E292 a and stage 1 E373). A replaced champion stays in `champions/` as a comparison network: its code is in its tag, it is evaluated again from a worktree of that tag.

## 8. Porting: what is kept, packed, deleted
| | keep (`D:\...\data`, `runs`) | pack (`D:\...\archive`, in the catalog) | delete (after the catalog, by the clean-up verb) |
|---|---|---|---|
| A champion and its chain | weights, predictions, config, log | - | - |
| B comparison networks | weights, predictions | - | - |
| C all other runs | - | `run.json`, cards, logs (text); checkpoints (hard rule 5, see decision 4) | caches, pictures, videos, predictions |
| data | raw nuScenes; ground truth of the current signature; caches that A and B read | - | older ground-truth versions (D40), caches that nothing in A and B reads |
| code and records | the code of A, moved | everything else, in the freeze tag | from `main` and the disk, after the freeze tag is verified |
| run tags (1,343 local) | tags of A and B, the freeze tag | a git bundle of all of them | the other local tags, after the bundle is verified |

Sizes on 2026-11-26 (not in git; links not followed): lane folder 49 GB (weights 9.4, caches 38.1, pictures and documents 1.2); BEV folder 123 GB (37.1, 80.8, 4.7); `~/exp_gcar_b123` 161 GB in the WSL image; on D: raw nuScenes 121 GB, `NuScenes_cache` 71 GB. D: has 340 GB free: moving everything (333 GB) would not fit with a margin, so derived data are thinned first, then the rest moves. C: has 45 GB free and gains most of the 333 GB.

**Champions are defined in several places today, and they disagree.** Lane: handover, `PROBLEM.md` and the winner folder name E372 S4 (D120); the protected entry of `models/registry.json` still names the builder of loop 16; `keep3.json` lists three networks of that old family. BEV: the winner folder names `t21_ctl`, `records/champion.json` names `t8_ctl` and is called stale; the winner folder's code does not run any more, its numbers are on ground truth v8, its weights load under today's code. So class A takes today's code, not a winner folder.

**Texts.** `PROBLEM.md` and `DESIGN.md` are carried over and trimmed (the technical rules moved from the decisions into `DESIGN.md` with D131). A decision belongs to its problem: each problem has its own `DECISIONS.md`, the root file holds only what concerns more than one; the ids stay `D<n>` with one shared count. A new file starts with the old permissions the owner confirms one by one; the old file stays in the tag. BEV has no file of its own today (its `project.json` points into the lane's), no `PROBLEM.md`, `HANDOVER.md` or `CLAUDE.md` (written from `plan/` and its README), and the TEST log its `project.json` names does not exist. The lane TEST log is carried over complete: which design has seen TEST does not expire.

## 9. Steps
| step | what | done when |
|---|---|---|
| S0 structure | `problems/`, `champions/` with their READMEs; the toy moved to `problems/toy/` | done 2026-11-26; the toy selftest passes there |
| S1 core | the new core (`research/`, `report/`, evaluation step, judge, adopt, machine file with the two roots, holdout-aware reader, step graph, kill rules in the worker, one test command), the fixes of section 1; the toy rebuilt on it, its runs into `D:\...\runs\toy\`; the old tools stay until S6 | system selftest; the toy refuses a bad control, a holdout read and an edited guard |
| S2 quiet point, catalog, freeze | every chat stops writing; `catalog.csv` of all old experiments and runs (id, question, numbers with split, verdict, by tool or hand, commit, checkpoint hash, class A / B / C); tag `archive/pre-restructure` with the untracked records and scripts; tag bundle and checkpoint pack on D: | a worktree of the freeze tag runs today's tests; pack and bundle hashes checked |
| S3 data to D: | the speed check; derived data of class C thinned; data and runs moved by the plan of section 6; `datasets/nuscenes/`; the machine file on this PC | kept files' hashes unchanged; links rebuilt; a training epoch as fast as before, or the owner's OK |
| S4 first problem (BEV) | `champions/bev_fusion/`; the class A code by `git mv` (history stays), its glue replaced by the contract; anchor test; the champion report rebuilt from stored predictions | anchor bit for bit; report regenerated |
| S5 lane | the same; its own trainer moves with it (`trainkit` later, as an experiment); `~/exp_gcar_b123` under a manifest with its producers | the same |
| S6 retire | `git rm` of the old problem folders, `reporter/` and the old tools of section 10; the untracked rest removed after the catalog check (links listed and removed alone first); root and problem `CLAUDE.md`, READMEs rewritten; root `DECISIONS.md` with the switch | `check_structure` clean; every file has a README row |
| S7 go on | new experiments number on (lane from E396, BEV from t27) | the first one judged by the tool |

**Order.** BEV has a runner, an adapter and `trainkit` and is the smaller step. The lane has none of these but has first call on windows and chat time (D57). Proposal: core, then BEV, then lane; the other order is the owner's right. Ranks and data shards (P0.6) come after the first real problem runs.

## 10. Deprecated and dead code
A static scan of the tracked `.py` and `.sh` files on 2026-11-26: imports, launched commands, run records; nothing was executed. Blind spots: scripts found by bare name through `_py14.sh`, modules loaded by path, code named in configs. So "reached by nothing" means: no code and no record points at it.

**General code**
| class | what |
|---|---|
| live, goes into the new core | `run.py` (the launch part), `ledger`, `metrics` (become `index`), `experiment`, `card`, `decide`, `result` (become `judge`), `provenance`, `testgate`, `guard_hook`, `hold`, `machine_lock`, `project`, `gitfast`, `recipe`, `check_git`, `commit_paths`, `archdraw`, `install_agents`, `clean_temp`; `Training`, `trainkit`, `Evaluation`, `suitecli`, `reporter`; agents, schemas, templates |
| old records only, leave `main` at S6 | `package`, `keep3`, `schedule`, `run_report`, `topic_table`, `check_numbers` (`reporter check` does it), `jobrunner/`, Training's `queue_api` |
| dead now | `plan.py`, `lineage.py` (only their own selftests use them), `reporter/plots.py` |

**Problem code** (files / thousand lines)
| | lane | BEV |
|---|---|---|
| tracked `.py` and `.sh` | 673 / 128.5 | 349 / 76.7 |
| needed by the champion and its stages (class A) | 108 / 22.5 | 77 / 18.7 |
| E388 - E395 scripts; BEV extras (CRN, SOA tables, streams) | 22 / 7.6 | 17 / 5.0 |
| the old chain of loops 8 - 18 (only via `evaluate.py`, `lane_suite.py`) | 34 / 7.1 | - |
| interfaces, figures | 11 / 2.7 | - |
| reached only by tests | 103 / 22.8 | 57 / 12.7 |
| reached by nothing | 395 / 65.9 (337 in `experiments/`, 285 of them recipe copies; `reports/build`; 20 data makers) | 198 / 40.2 (`studies/`, `training/legacy`, experiment snapshots) |

About a sixth of the lane code and a quarter of the BEV code moves; the rest stays in the freeze tag.

**Findings that change the move**
- The lane champion trains with its own `training/trainer.py`, BEV with `trainkit`.
- The lane champion reads `~/exp_gcar_b123` (161 GB, 52 lane files name it) next to `data/autogt`; for some of it no producer was found (e.g. `runs/refiner_stats_dec32_logit7.npz`).
- The protected champion entry of `models/registry.json` still names the loop-16 builder, and `evaluate.py`, the documented front door, enters that old chain.
- A broken import: `training/train/p17_gate.py:19` needs `p13_temporal_val`, which is gone. 12 scripts launched since 2026-11-18 are only in run tags.
- Duplicates the core replaces: about ten ways to launch or queue, two GPU gates, two trainer loops, four report builders, two number checks, about seven private bootstraps, about seven timing tools, three ground-truth stores, four runners.

## 11. Clean-up rules
1. Nothing leaves the tree before the freeze tag exists with the untracked records in it and a fresh worktree of it runs today's tests.
2. The new core replaces the old tools; there are never two launchers or two gates in use. Dead code leaves `main` by commits; the tag keeps it.
3. Raw data, checkpoints and untagged scratch are never deleted (hard rule 5): checkpoints move to `runs\` or the pack, scratch goes into the freeze tag first. Links are listed and removed alone before any folder removal.
4. Derived data are deleted only by the clean-up verb after the catalog lists them.
5. Every step ends with the system selftest and, for a problem, its anchor numbers.
6. Protected files and the hook change only under a decision of the owner.

## 12. Decisions
| # | decision | state |
|---|---|---|
| 1 | three classes (A code moves, B facts and predictions, C tag and pack only); nothing old is converted | proposed |
| 2 | `problems/lane_center`, `problems/bev_fusion`, `champions/`; no old folders, git is the archive; decisions per problem | the owner, 2026-11-26 |
| 3 | code and data apart: data and runs on D:, code on C:; champion reproducible through git, intermediate results not; reports regenerated from predictions and raw data by code in git | the owner, 2026-11-26 |
| 4 | old checkpoints of class C: packed on D: and kept (hard rule 5 as it is), or deleted beyond the champions' chains (changes hard rule 5) | open |
| 5 | BEV champion `t21_ctl` (best by value) or `t20_base_s0` (the D48 reference); comparison networks; score and guard bars, ground-truth version | lane: E372 S4 with `topo_F1` and guards (answered earlier); BEV open |
| 6 | order: BEV first, or lane first (D57) | proposed: BEV first |
| 7 | the winner folders end (changes D139 / D140) | follows from 2 and 3 |
| 8 | a frozen rule script counts as a rule written before the run (hard rule 4) | proposed |
| 9 | one decision row for the switch: the hook looks at `problems/*/` and `champions/`, the TEST check also in the core reader, the protected lists, the rule texts that name `run.py` and `package.py` | at S6, when S1 is green |
| 10 | if WSL reads D: too slowly: a Linux disk file on D: mounted into WSL | after the speed check |

## 13. Measured, kept, deleted (2026-11-27, D155)
**Measured through the research system** (VAL; the records are in `problems/lane_center/experiments/`):
| what | result |
|---|---|
| lane E396 (kind measure): the champion E372 S4, seeds 0 and 1, `research start` from its snapshot (dump step + evaluation step) | topo F1 0.668172 / 0.681281, junction F1 0.2538 / 0.2481, Q 0.6796 / 0.6808: the numbers of its v1 cards to the last digit; registered as the champion under D120 (`champions/lane_center/`) |
| lane E397 (compare): keep threshold 0.5 against 0.6, the control reused from E396 (`from`) | TIE: gain -0.0026, paired scene bootstrap [-0.0039, -0.0017], band 0.01; the guard lanes_found breaks (-0.0123); the champion's decode stays |
| BEV E27 / E28 (kind measure; the weights imported, the warm evaluation on ground truth v13, `spread:1000:val`) | `t21_ctl` board7 0.4631 / 0.4635 (seed 0 / 1), `t20_base_s0` 0.4569; 0.006 apart, inside the band 0.02; the older 0.4549 / 0.4465 are cold reads on ground truth v8 and not comparable; the champion stays the owner's choice |

**Kept.** Raw nuScenes (`D:\Auto-research\NuScenes`, 121 GB, never deleted). In the data root `D:\Auto-research\data`: `lane_center/b123`
(30.1 GB: the B' features of train and val, the val rows hold the 833 TEST rows too, graph GT, poses, PCA), `groundtruth/lane/{9ad1d4ce30,
8a04143278}` (the l6 stores), `groundtruth/a107e0b4e8` (front v13, from C:), `groundtruth/cache` (the reader's tables), `bev_fusion/` (the
caches the BEV champion reads: GT v13, camera features, BEVFusion features, radar, front images, the annotation table; the pretrained DualBEV
and BEVFusion weights), `nuscenes/` (the view: links to the raw data and the CRN info files). In the runs root `D:\Auto-research\runs`:
`lane_center/` (the champion chain E292 a, E273 p, E271 a, E373, E372 S4 with seeds 0 and 1, the caches E365 and E357, the refiner
statistics, 6.9 GB; the runs of E396 and E397) and `bev_fusion/` (`t21_ctl_s0`, `t21_ctl_s1`, `t20_base_s0`). In WSL: `~/work`
(GaussianCaR code and weights, the BEV comparison network) and the conda environments. `D:\Auto-research\archive\lcr_local_tags_2026-11-24.bundle`
(25 MB, the code of the old run tags).

**Deleted** (listed here before; nothing of it is read by the code of `v2`):
| where | what | size | why |
|---|---|---|---|
| WSL `~/exp_gcar_b123` | the old lane pipeline's features, datasets and failure series; the 30 GB the chain reads were copied to `data/lane_center/b123` first | 164.8 GB | old experiment data |
| WSL `~/exp_gcar_b456`, `~/exp_crn_b123`, `~/exp_crn_mini`, `~/exp_gcar_mini`, `~/exp_gcar_b123x` | old feature extractions (GaussianCaR, CRN) | 80.4 GB | old experiment data |
| WSL `~/mutk`, `~/mutk2`, `~/head_check`, `~/exp_gcar_gtchk`, `~/_verify_v4` - `_v7`, `~/_review_v6_b`, `_c`, `~/critic_fixes_tmp` | scratch of old checks | 6.9 GB | scratch |
| D: `lane_archive_2026-11-01` | the heavy files of the lane runs E19 - E72 | 19.9 GB | old experiment data |
| D: `NuScenes_checkpoints/bev_fusion` | superseded BEV checkpoints | 6.7 GB | no champion chain |
| D: `NuScenes_cache/bev_fusion/cam_feats_parts7_10` | feature shards of the 340 scenes of parts 7 - 10 (only 700-scene runs read them) | 10.4 GB | a cache nothing kept reads |
| D: `backup_best_network_2026-11-10/t21_ctl_s0` | a copy of `runs/bev_fusion/t21_ctl_s0` (the shared files checked identical; its code snapshot and notes moved into the kept run as `backup_2026-11-10/`) | 0.4 GB | duplicate |
| D: `NuScenesAdditionalGt/c643a7539d`, `lane/{0702a065c6,3576fa1f7e,db288acca5}`, `examples`, `reports` | front GT v8, older lane GT, generated pictures | 3.8 GB | older GT versions (D40) |
| C: `bev_fusion/`, `bev_fusion_lane_center/` (untracked rest), `data/autogt` (links; v13 moved first), `temp/` | the old problems' data, runs, results and scratch | about 178 GB | after the merge into `main`: the folders leave the tree with the merge, their untracked rest after it |

Done on 2026-11-27 for WSL and D: (`D:\Auto-research` now holds `NuScenes`, `data`, `runs`, `archive`; D: 271 -> 311 GB free); the
reader's table cache moved to `data/groundtruth/cache`, the front store v13 was copied to `data/groundtruth/a107e0b4e8` (34,150 files, bytes
equal). The rows on C: wait for the merge. The WSL disk image does not shrink by itself after a deletion: compacting it (`wsl --shutdown`, then `Optimize-VHD` or `diskpart compact vdisk`
on `wsl\ext4.vhdx`) is the owner's step.
