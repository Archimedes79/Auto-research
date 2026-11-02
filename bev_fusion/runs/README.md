# runs - one folder per training run (not in git except this file)

`runs/<name>/` holds what `temp/train/run_cfg.sh` / `training/train.py` write: `config.json`, `log.jsonl`, `train.log`, checkpoints
(`ckpt_*.pt`, older ones moved to `data/checkpoints/<name>/` = `D:\Auto-research\NuScenes_checkpoints\bev_fusion\<name>\`, see `MOVED.txt`), `code/` (a `git archive` snapshot of
`training/` at the pinned commit, `SNAPSHOT.txt` names it; `evaluation/bev_adapter.py` evaluates a run with its own snapshot) and
`eval/` (metrics, tables, scene records, pictures, `suite/` = the evalsuite stores). `.done` marks exit code 0.

Names: `t1_*` first whole-chain trainings, `t2_*` block swaps, `t3_*` the improve loop, `t4_*` the SOA comparison; `_golden_*` a copy (config + eval) of a run used for the golden
numbers of the evaluation migration (`../evaluation/golden/`); `smoke_*` / `distill_*` short tests. Results of a run are summarised in `../experiments/<id>/`.
