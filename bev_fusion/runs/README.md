# runs - one folder per training run (not in git except this file)

`runs/<name>/` holds what `tools/run_ar.sh` (before D23 the superseded `studies/recipes/train/run_cfg.sh`) / `training/train.py` write: `config.json`, `log.jsonl`, `train.log`, checkpoints
(`ckpt_*.pt`, older ones moved to `data/checkpoints/<name>/` = `D:\Auto-research\NuScenes_checkpoints\bev_fusion\<name>\`, see `MOVED.txt`), `code/` (a `git archive` snapshot of
`training/` at the pinned commit, `SNAPSHOT.txt` names it; `evaluation/bev_adapter.py` evaluates a run with its own snapshot) and
`eval/` (metrics, tables, scene records, pictures, `suite/` = the Evaluation stores). `.done` marks exit code 0.

Names: `t1_*` first whole-chain trainings, `t2_*` block swaps, `t3_*` the improve loop, `t4_*` the SOA comparison; `_golden_*` a copy (config + eval) of a run used for the golden
numbers of the evaluation migration (`../evaluation/golden/`); `smoke_*` / `distill_*` short tests. Results of a run are summarised in `../experiments/<id>/`.

**Runs made before D76 are not run any more (D81, Mirko 2026-11-10: "Ich brauche die alten Läufe nicht mehr laufen lassen, alles gut").** Their `code/` snapshot says `from autogt...` (the generator was renamed GtGenerator, D76; the alias was removed, D78), so no evaluation, Runner or resume loads it. Nothing is deleted: the folders, the checkpoints (a warm start `init_from` reads only the weights and works under the live code) and the best three networks (`../records/keep3.json`) stay. The reference run of the real-data gates is `chain2_t20` (30 steps, made after D76 by the acceptance chain, `../records/jobs/chain/README.md`). If an old run must run again, the way back is one small alias module (see D81).
