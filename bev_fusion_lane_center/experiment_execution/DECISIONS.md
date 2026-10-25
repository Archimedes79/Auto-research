# Decisions for Mirko

The only channel for decisions that are yours (agreed 2026-10-25): **promoting a champion** (with its one TEST check), **changes of the problem** (`PROBLEM.md`: goal, metrics, gap ranking, constraints), **changes of the tools or the gates** (`autoresearch/tools/`, the rules of `autoresearch/METHOD.md`), and **permissions** (downloads, installs, mail, shutdown, backups, remote, renames). An agent writes the entry (question, options, recommendation, evidence with numbers from metric cards, what each option means) and does nothing about it until the answer is recorded here with a date. `HANDOVER.md` points to the open entries; every loop summary ends with them.

## Open

| id | question | options | recommendation | evidence |
|---|---|---|---|---|
| D3 | rename the workspace folder `C:\ML-Experiment` -> `C:\Auto-research-lmc`? | yes (with the prepared script) / no | yes, if you want the name; nothing depends on the old name any more | prepared by another session: quit the Claude app, run `bev_fusion_lane_center/temp/rename_repo_folder.ps1` (shuts WSL down, renames, re-registers the WSL disk, copies the Claude memory), reopen in the new folder, run `tools/maintenance/run_all_tests.sh` |
| D4 | promote a fast variant for deployment (champion decision; its one TEST check comes with it)? | `fast` j1 + pruning (17.5 ms, Q 0.728) / `fast_recall` j1rc (20-22 ms, Q 0.739, lanes found 0.736; Q -0.038 with three cameras off) / j1eq (17-19 ms, Q 0.735, safer under camera loss) / keep `champion` f1 + time recursion (24 ms, Q 0.735) | wait for loop 19's stricter statistics (3 seeds + scene interval), then decide; j1eq is the safer fast candidate | VAL 873 frames, `evaluate.py` cards and the loop-18 report; j1rc / j1eq were never checked on TEST |
| D5 | purge `_to_delete/` and act on `tools/maintenance/CLEANUP_LIST.md`? | purge now / later | after D3 (if the folder is renamed, purge afterwards) | `_to_delete/README.md` lists what is staged and why |
| D6 | start the roadmap with M0 + M1 (golden numbers, record and provenance hygiene in the current scripts)? | yes / discuss first | yes; it runs in parallel with the cycle-19 research (P1 / P2 touch no code), loop 19's GPU runs wait for it | `autoresearch/ROADMAP.md` |
| D9 | fix the launcher's git snapshot: refuse to start when git has no user identity, write git's own error into the run record, and test that case in `selftest.py`? (a tool change) | yes, as part of M1 / the D8 order / later | yes, with M1 (it is record hygiene) | 2026-10-25: E234.tr0 / tr1 recorded `commit: git-error` because the WSL git had no identity (the repository was made by the Windows git; the selftest builds its own repository and sets one, so it never saw this). Fixed for now by setting user.name / email in the repository's own config; the two runs were tagged by hand at the commit they started from (07a4bad) |

## Decided

| date | decision | consequence |
|---|---|---|
| 2026-10-25 | D8: (a) the autonomy core before the package facades (Mirko in chat 14:05: "D8: (a) yes. Record it in DECISIONS.md and start with M0 golden numbers, then P5 and P1 as proposed.") | order M0 -> P5 -> P1 -> P2 -> P4 -> P3 -> P6-P8, M3-M5 after loop 19 (`autoresearch/REVIEW_2026-10-25_loop18_lead.md`); the loop-18 lead chat owns the tool work (HANDOVER names it) |
| 2026-10-25 | D1: download the nuScenes radar parts 4-6, then plan the feature extraction for the 255 new scenes | done 12:57: three parts (632 MB) from the public bucket `motional-nuscenes`, each verified against its S3 ETag, unpacked; camera and radar now on disk for 510 scenes (20,418 keyframes each); extraction plan `plans/data_extension_b456.md`; the use of the new scenes is D7 |
| 2026-10-25 | D7 option (b): keep TRAIN / VAL / TEST as they are; add TRAIN2 = TRAIN + 197 new official-train scenes (7,961 frames), CONFIRM = 39 new official-val scenes (1,567 frames, never used for tuning; confirms a champion candidate before its TEST check), UNSEEN = the 19 singapore-hollandvillage scenes (770 frames, a location not in b123); start the feature extraction steps 1-4 (new files only) | split file `data/splits_b456.json` (indices into `~/exp_gcar_b456/feats/{train,val}_tokens.json`); extraction runs E234.* in the ledger; plan `plans/data_extension_b456.md` |
| 2026-10-25 | git holds code AND all text records (no binaries, no data) | `.gitignore` whitelist extended: experiments (text), experiment_execution, run records, model configs and metric cards, `data/splits.json`, literature summaries, report digests |
| 2026-10-25 | single-file records (ledger, decisions, closed questions, plan) stay complete; agents read experiment folders of the last two loops in full, older loops through loop / weekly summaries | rule in `CLAUDE.md` and `autoresearch/METHOD.md` |
| 2026-10-25 | decisions come to Mirko through this file, not as file edits | champion promotion, problem changes, tool / gate changes, permissions |
| 2026-10-25 | TEST is not decided separately: one TEST check belongs to a champion promotion, for the candidate and the current champion only | the guard in the tools follows when M1 / M7 are built |
| 2026-10-25 | the outer loop (research cycle P1 / P2) is started by hand, step by step; the inner loop P3 runs autonomously | `autoresearch/prompts/research_cycle.md` to be split into steps |
| 2026-10-25 | statistics: 2 seeds for screening, 3 seeds + scene-bootstrap interval before a champion proposal | `decide` rule in M1 / M6 |
| 2026-10-25 | no VAL split now (data is limited); see D1 | |
| 2026-10-25 | one lead chat per loop owns the code; other chats work in `literature/` or `reports/` or read; `HANDOVER.md` names the owner | |
| 2026-10-25 | the safety work (M0-M2) and the cycle-19 research run in parallel; loop 19's GPU runs wait for the safety work | |
| 2026-10-25 | D2 GitHub repository: private `Archimedes79/Auto-research-lmc` (created empty by Mirko); `origin` set, `main` pushed (5 commits, tip af34624) | code and text records are on GitHub; later pushes only when Mirko asks; the remote does not change with the folder rename (D3) |
