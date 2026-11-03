# trainkit - the project-independent training engine

The parts of a training that do not depend on the network, the loss or the data: EMA, the learning-rate schedule with its own rates per module, clipping and the optimiser step with the non-finite rules, AMP with the GradScaler, atomic checkpoints with resume and warm start, the log files, and the stop protocol with the trainsuite. Plain torch, no project code; the lane project has a trainer of its own and does not use it. The BEV system (`bev_fusion/training/train.py`) is the first user (D48); the equivalence test `bev_fusion/tests/test_trainkit_equivalence.py` runs the trainer from before the move and the live one on the same toy and requires the same losses, weights, files and messages.

| file | content |
|---|---|
| `config.py` | `Config`: the fields the engine reads (steps, lr, warm-up, accum, clip, EMA, AMP, cadence of logging and saving, seed, warm start, device) and the rules of a resume; a project extends it |
| `ema.py` | `EMA` (weights and buffers, decay ramp, frozen keys stay the value itself), `ema.applied(model)` for the validation |
| `schedule.py` | `lr_factor` (warm-up from lr / 3, cosine), `scheduled_lr` (an `lr_groups` prefix has its own rate and warm-up) |
| `optim.py` | `param_groups` (decay / no decay, one pair per `lr_groups` prefix), `clip_groups` / `clip_gradients` (global or per module group), `optimizer_step` (clip, step, skip a non-finite gradient, fp16 overflow), `bad_steps_after`, `poisoned_buffers` |
| `amp.py` | `amp_context`, `make_scaler` (GradScaler for fp16), `scaler_extra` |
| `checkpoint.py` | `save_checkpoint` / `load_checkpoint` (state without the `external` prefixes, optimiser, EMA, RNG), `resume` (newest readable, a damaged one is kept as `.damaged`), `check_resume` (a resume must be the same configuration), `refuse_slim_resume`, `warm_start`, `prune_checkpoints` |
| `runlog.py` | `supersede_log` (log rows above the resumed step go to `log_superseded.jsonl`), `val_logged` |
| `control.py` | the stop protocol: `CONTROL` / `STOPPED` files and exit code 75 (standard library only; `trainsuite/control.py` re-exports it) |
| `util.py` | `seed_all`, `resolve_device`, `prepare_run_dir`, `detach_tree` |
