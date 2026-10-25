# training - the system (models, data, losses, training, inference)

Long-lived code of the lane-graph chain. All modules are imported flat (`import p16_builder as PB`); the launcher `tools/env/_py14.sh` puts every folder on `PYTHONPATH`, so a module can move between the folders below without breaking imports. **New model code goes here; one-off analysis does not (that is `temp/`).**

| folder | role | files (main ones) |
|---|---|---|
| `trainer.py` | **general trainer** for new work: AdamW + warm-up / one-cycle, AMP, clipping, checkpoint every N steps with RNG state, `resume=True`, NaN skip, loss-spike roll-back, `done` flag only after success. Self-test: `trainer_selftest.py` (crash at step 120 + resume gives the same weights as an uninterrupted run). | the old scripts below carry their own copies of this boilerplate |
| `models/` | network definitions | `p16_builder.py` (loop-16 builder D1-D8: lattice seeds, teardrop sampling, survival read-out, children, rescoring, junction head), `p16_geom.py`, `p16_tokens.py` (vertex tokens for the route head), `p16_time.py` (time recursion), `p16_ref.py` (numpy reference of the spec), `p10_net.py` (+ `p10_expert.py`, `p10_wave.py`: trace network E94), `p13_jhead.py`, `p13_junction.py`, `p13_phead.py` (junction / piece heads), `p15_refine.py` (refinement decoder E120), `p8_assign.py`, `p8_seg.py` (reference system A) |
| `data/` | datasets, constants, libraries | `p16_data.py` (BuilderData: cached features, ground truth, previous-frame priors), `p16_common.py` (paths, split sizes, `val_subset`, card writer), `p10_data.py`, `p14_chain_data.py` / `p16_chain_data.py` (chain data for the consumer: tokens + route), `p8_lib.py`, `sd_lib.py` (SD map), `assoc_lib.py`; **`prep/`** = builders of the cached data (feature extraction, ground truth, poses, priors, splits) |
| `losses/` | training losses | `p16_loss.py` (matching-free losses of the builder) |
| `train/` | training entry points | `p16_train.py` (builder, all designs), `p16_ctrl.py` (control arm E94 + E120), `p10_train.py`, `p10_route_train.py` (route head), `p14_chain_consumer.py` (planner consumer ConsB, train + eval), `dense_head.py` (grid head: train + eval with `evaluation/lane_grid`) |
| `pipeline/` | inference and assembly | `p16_infer.py` (path-union assembly, junction anchoring, fork relocation, `FastUnion`), `p16_runtime.py` (CUDA-graph `FastBuilder`), `p16_pipeline.py` (builder -> route head -> consumer chain), `p16_tok_resample.py` (route head asked along the edges), `p16_ensemble.py`, `p16_hyst.py`, `p10_infer.py`, `graph_from_dense.py` (dense map -> graph, rule system), `line_tracer.py` |
| `search/` | experiment search harness | `p16_triage.py` (successive halving: queue, rungs, cuts, writes `experiment_execution/indexes/TRIAGE_STATE.md`) |
| `legacy/` | old modules of loops 2-8 that live code or the code of old experiments (`experiments/E*/code/`) still imports (refiner, temporal_b, route_head, eval_pert_val, p13_temporal_val, lane_former, ...). Not for new work. Audit and plan for consolidation: `tools/maintenance/CLEANUP_LIST.md` section C |

Loop 17 (`p17_*`: failure series, gate, map-conditioned builder, joint fine-tune, consumer under failure) belongs into `models/`, `train/` and `evaluation/loops/loop17/` and `evaluation/drivers/` (restored on 2026-10-24 after a rollback deleted them; the replay is in `experiment_execution/loop17_recovery/`).

## Ground rules
- Everything that trains must be resumable and must write `log.txt`, `status.json`, `model.pt` into `runs/<name>/` (use `trainer.fit`; retrofit `p17_mapft.py` first - its crash at 3500 / 4800 steps lost 2 h).
- Metrics are not defined here - call `evaluation/` (`sparse_lanes`, `lane_grid`).
- Cached foundation features are read from the WSL home (`~/exp_gcar_b123`, `data/README.md`); nothing here writes there except `prep/`.
- Speed is a criterion: a change of the pipeline keeps the graphs identical (`evaluation/drivers/p16_fast_check.py`).
