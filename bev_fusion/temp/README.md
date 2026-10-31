# temp - scripts of the studies and of the setup (scratch, but some of it is used)

Rule of the repository: scratch goes here, a reusable tool is promoted to a proper folder when it is used a second time; nothing in
`training/`, `evaluation/`, `autogt/` imports this folder. The review of 2026-10-30 found that the rule is not kept for the items marked
**used**: documents and launchers point into this folder. They are the candidates for promotion (see `../README.md`, review notes in
`review/`).

| folder | what | class |
|---|---|---|
| `train/` | `run_cfg.sh` (train one config from a `git archive` snapshot of `training/` at a pinned commit, `.done` only on exit 0), report builders (`plot_curves`, `build_tables`, `stack_images*`, `draw_architecture*`), cache fill scripts | **used** (the launcher of every run) |
| `study/` | the queue drivers `lane.sh` / `lane2.sh` / `lane3.sh` (run a list of configs one after another), `queue*.txt`, result collectors (`collect*.py`, cited by the t3 report) | **used** (a queue is running there) |
| `setup/` | cache builders, `get_dualbev_ckpt.sh`, one-time table merges | **used** (`data/README.md`, `models/README.md` name the build / download scripts) |
| `data/` | download scripts of the nuScenes parts 7-10 | **used** (`data/README.md`) |
| `radar/` | radar statistics and the reconstruction test `radar_recon_test.py` | `radar_recon_test.py` is **imported by `tests/`** (with the cache `temp/radar/cache/`) |
| `checks/` | single-purpose checks, queue chains (`chain_*.sh`), probes, plots | scratch (one-off scripts that were finished moved to `../_to_delete/temp_oneoffs/`; on 2026-10-31 also the t1-era queue chains `chain_a` ... `chain_x` and job probes, and 15 unreferenced one-offs of `train/` and `setup/`) |
| `calib/`, `audit/`, `timing/` | code of the t1 / t2 / t3 experiments (calibration, audit of the chain, latency) | experiment code, cited from `../experiments/*/experiment.md` |
| `soa_gap/`, `t4/`, `t5/`, `t7/` | code and results of the experiments t6, t4, t5, t7 | experiment code, cited from `../experiments/` |
| `scout_tables/` | the raw tables of the literature scouts | input of `../literature/papers.csv` (not scratch) |
| `review/` | probes and notes of the code reviews (what the pictures of a run showed, apply scripts of the structure review) | records |
| `cache/` | generated caches of the scripts | not in git |
