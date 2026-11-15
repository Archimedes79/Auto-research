# t26 G0 records (2026-11-15): the base against the references, kept as text

- `scenes43.txt`: the 43 val scenes in which GaussianCaR's logits and CRN's inputs exist (1620 scoring frames); `bev_suite.py eval --scenes $(cat records/t26_g0/scenes43.txt)`.
- `ref_maps_base_out.md` / `.json`: `soa_maps.py` of the base (`runs/t20_base_s0` ckpt 6000 EMA, variant folder `temp/t26/base_run`, GT v13, `--allow-gt-mismatch`): the tables and the per-row numbers with the scene bootstrap; `--frames-from ref_maps_base_out.json` makes a later scoring use exactly these frames. The per-frame counts (`out_frames.npz`, 1.7 MB) stay in `temp/t26/ref_maps_base/`.
- The metric cards of the base evaluations (0.8 MB each) are not kept here: `runs/t26_base/eval/card.json` (with the velocity rows) is made when `t26.base_eval_vel` has ended, see the card's launch gate 8.
