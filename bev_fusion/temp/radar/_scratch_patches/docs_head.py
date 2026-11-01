"""applies the doc / script rows of the radar-slot wiring to the HEAD copies in headcopy/ (so other chats' uncommitted edits stay out)."""
H = "temp/radar/_scratch_patches/headcopy/"


def edit(name, old, new):
    p = H + name
    t = open(p, encoding="utf-8").read()
    assert t.count(old) == 1, (name, old[:50], t.count(old))
    open(p, "w", encoding="utf-8").write(t.replace(old, new, 1))


edit("tests_README.md", "| `test_object_head.py`",
     "| `test_radar_slot.py` | the radar slot `delta_slots` (`training/radar_encoder.py::RadarEncoder`) inside the whole chain, nuScenes mini, CPU, camera off: "
     "check_config refuses c_rad < 61; the model returns `radar_side` (features + lossless raster) and passes the output contract; "
     "`forward_loss` with `radar_aux` > 0 adds weight x side loss (weight = `aux_weight(step_frac)`), the gradient reaches the encoder, "
     "the side-loss decoder and the trunk; `radar_aux` = 0 adds nothing; validation logs `radar_kept` and no side loss; radar_kind raster is "
     "untouched (no `radar_side`, identical loss) | all eight checks | passes (2026-11-01) |\n| `test_object_head.py`")
edit("run_gates.sh", 'run "radar encoder"     "$PY" training/radar_bev.py\n',
     'run "radar encoder"     "$PY" training/radar_bev.py\nrun "radar encoder, lossless" "$PY" training/radar_encoder.py     # delta-coded slot raster: exact round trip, side loss, monitor\n')
edit("run_gates.sh", '  run "trainer"         "$PY" training/train.py',
     '  run "radar slot"      "$PY" tests/test_radar_slot.py              # delta_slots in the whole chain: side loss, gradients, old path unchanged; about 2 min\n'
     '  run "trainer"         "$PY" training/train.py')
edit("training_README.md", "`forward(..., with_state=True)` = the lossless raster as explicit stream;",
     "`forward(..., with_state=True)` = the lossless raster as explicit stream; in the chain it is the radar slot `radar_kind: delta_slots` (`model.py`, needs `c_rad` >= 61) "
     "and `TrainConfig.radar_aux` / `radar_aux_floor` / `radar_aux_half` switch on and decay its side loss (`train.forward_loss`; `tests/test_radar_slot.py`);")
edit("configs_README.md", "| `t6_maps_only.json`",
     "| `r2_rad_raster32_s0/s1`, `r2_rad_raster64_s0/s1`, `r2_rad_slots64_s0/s1` | experiment r2_radar_slot (run ids `r2.*`): radar-only (camera off) comparison of the radar slot, "
     "`t3_radar_only` with the t11 velocity recipe (`object_moving`, accum 2, 1500 steps, same 9000 samples): shipped raster block (c_rad 32), width-matched raster (c_rad 64), "
     "`delta_slots` (c_rad 64, side loss `radar_aux` 1.0 -> 0.05); two seeds each | 2026-11-01 |\n| `t6_maps_only.json`")
print("docs ok")
