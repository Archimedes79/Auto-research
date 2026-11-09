"""scratch: put the t16 rows into the HEAD copies of two READMEs (each insertion must match exactly once)"""
T = "bev_fusion/temp/t16/"
row_exp = ("| `t16_full_network/` | does the complete chain (dual camera, radar delta slots + radar memory, motion block, velocity hurdle; five-layer framework, GT v8) train, "
           "and what is its baseline? bring-up with gates: small set (autogt small, 10 scenes) -> timing -> module probes -> fixes -> large set (428 / 82 scenes) + matched control | draft, no run yet (D44) |\n")
s = open(T + "experiments_README.md", encoding="utf-8").read()
key = "| `t13_complete_network/`"
assert s.count(key) == 1
open(T + "experiments_README.md", "w", encoding="utf-8").write(s.replace(key, row_exp + key))
row_cfg = ("| `t16_full.json`, `presets/small_set.json` | experiment t16_full_network (`../../experiments/t16_full_network/`): the complete chain in one config (`t10_dual_long_s0`'s recipe, 6000 steps, accumulation 2, + radar delta slots R with `radar_aux`, radar memory with the Doppler static gate, motion block M joint with `box_vel_from_field`, `probe` on); `small_set` is the preset that points it at autogt's small set (10 scenes of nuScenes mini, features linked in `data/cache/small/`): `python -m trainsuite start ../bev_fusion/training/configs/t16_full.json --id t16.small_s0 --preset small --preset small_set --no-card` | written 2026-11-02, no run yet |\n")
s = open(T + "configs_README.md", encoding="utf-8").read()
key = "| `t14_dc_ctl_s0.json`, `t14_dc_fusion_s0.json` |"
assert s.count(key) == 1
open(T + "configs_README.md", "w", encoding="utf-8").write(s.replace(key, row_cfg + key))
print("rows added")
