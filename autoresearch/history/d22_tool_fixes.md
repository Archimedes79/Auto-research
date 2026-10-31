# D22 - six small fixes of the research tools (staged, NOT applied)

Source: the structure review of 2026-10-30 (`bev_fusion/temp/review/STRUCTURE_REVIEW_2026-10-30.md`); every item below was re-read in
the code on 2026-10-31 (line numbers of that day). Since D21 the tools are not protected: applying these needs no grant, but
`python3 autoresearch/tools/selftest.py` (throw-away project) and an `ar-reviewer` pass (mode code) before they are used. Each fix
is a few lines; each gets a selftest case (named below). Nothing here changes a ledger number, a decision rule or a stored run.

| # | file | what happens today (verified) | change | selftest case |
|---|---|---|---|---|
| 1 | `tools/clean_temp.py` `run()` (line ~76-80) | `git_snapshot` may return `("no-git" / "git-error", "", note)` (run.py:102-137: no repository, `read-tree` / `add` / `commit-tree` failed); the code prints it and `shutil.rmtree`s the scratch anyway. The snapshot covers only `cfg["code_dirs"]`: a problem whose `code_dirs` lacks the scratch folder (the lane has `temp` in it, a new problem may not) loses the scripts for good | with `--delete`: refuse (return 1, print why) when `tag` is empty or `cfg["scratch"]` is not in `cfg["code_dirs"]`; `--move` stays allowed (nothing is lost) | `git=True` in a throw-away folder without a repository: `--delete` refuses, `--move` works |
| 2 | `tools/project.py` `root()` (line 31-36) | a relative `AR_PROJECT` is resolved against `autoresearch/`, a folder without `project.json` silently gets `DEFAULTS`, and `ledger.save` (ledger.py:35), `run.py:253/449`, `schedule.save` then `os.makedirs` a ghost problem folder with a ledger in it | after the path is resolved: `if not os.path.isfile(os.path.join(r, "project.json")): sys.exit(f"no project.json in {r} (AR_PROJECT={p!r}) ...")`; a relative `AR_PROJECT` is tried against the current folder first | `AR_PROJECT` pointing to an empty folder exits and creates nothing |
| 3 | `tools/run.py` lines 340, 482, 494, 509, 521, 551 | `run.json` is written with `json.dump(..., open(path, "w"))` while other tools poll it. In `GpuSlot.stale` (316-324) a torn read of the slot's `run.json` gives `record() == {}`, then `pid_alive(None)` is False, so a LIVE slot looks stale and is `rmtree`d and taken by a second job (3 jobs on 2 slots). The run-dir records are polled by `state_of`, `schedule derived` and `tick`: a torn read shows an empty status | one helper `save_rec(path, rec)` = `P.atomic_write(path, json.dumps(rec, indent=1))` (project.py:91 exists and fsyncs), used at the 6 places | two threads: one rewriting a record in a loop, one calling `record()`: never `{}` |
| 4 | `tools/run.py` line 505 | `(u is None or u < a.gpu_max_mb) and slot.try_take()`: when `--gpu-max-mb` is given and `nvidia-smi` fails (`gpu_used_mb()` returns None), the memory gate passes | `mem_ok = (not a.gpu_max_mb) or (u is not None and u < a.gpu_max_mb)`; the wait loop already prints that it waits | `gpu_used_mb` monkeypatched to None with `--gpu-max-mb 9000`: the job waits / ends `gpu_busy` |
| 5 | `tools/schedule.py` `change()` (line 83-86) | `add` appends without looking at existing items. `bev_fusion_lane_center/experiment_execution/schedule.csv` holds `E240.a3` and `E240.c3` twice (rows 10-11 `running`, rows 57-58 `planned`); `derived()` then shows the item from the first record only | `add` exits when `a.item` already exists; data fix: delete rows 57-58 of `schedule.csv` (the planned copies of two finished runs) | `add` of an existing item exits non-zero, the file is unchanged |
| 6 | `tools/ledger.py` `main()` (line 135-148) | `ledger.py note` / `add` without an id: `rows.append(dict(id=None, ...))` writes a ledger row with a blank id | `if a.cmd in ("note", "add") and not a.id: ap.error("id required")` | `note` without id exits, the ledger is unchanged |

Not part of D22 (separate decisions): machine-wide GPU slots (D19 is open: today `GpuSlot` lives under the problem's
`run_records/.gpu_slots`, a second problem gets its own 2 slots), splitting `run.py` (564 lines: card gate, holdout guard, git snapshot, slots,
ledger store, record state machine, supervisor), adopting `project.json` / `run.py` / ledger in `bev_fusion/`, the `LOWER` list of lane metric
names and the band 0.010 in `decide.py` (D18), the stale sentences in the CLAUDE.md files (below).

## Sentences that contradict D21 / the repository (instruction files: your edit or your OK)
- `bev_fusion_lane_center/CLAUDE.md` line 73 (Permissions): still lists `../autoresearch/tools/` and `../autoresearch/METHOD.md` as protected;
  `project.json` has them removed since D21 and holds `../autoresearch/tools/guard_hook.py` instead. Proposed text: "protected files (project.json "protected":
  PROBLEM.md, project.json, the TEST log, evaluation/golden.json, the champion / control entries of models/registry.json, the champion config card,
  ../autoresearch/tools/guard_hook.py, ../.claude/settings.json and settings.local.json, the grants file)".
- root `CLAUDE.md`, bullet `bev_fusion/`: "for now only the discussion ... later its own GT, evaluation and code". Today it has code (`training/`, `autogt/`, `evaluation/`),
  ~30 runs and 7 experiments. Proposed: "a second problem: camera + radar BEV system (plan, training, autogt, evaluation, experiments); not yet managed by `autoresearch/`
  (no project.json / ledger); see `bev_fusion/README.md`".
