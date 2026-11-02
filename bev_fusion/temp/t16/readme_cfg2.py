import subprocess
s = subprocess.run(["git", "show", "HEAD:bev_fusion/training/configs/README.md"], capture_output=True, text=True, check=True, encoding="utf-8").stdout
a = "+ radar delta slots R with `radar_aux`, radar memory with the Doppler static gate, motion block M joint with `box_vel_from_field`, `probe` on)"
assert s.count(a) == 1
b = "+ radar delta slots R with `radar_aux`, motion block M joint with `box_vel_from_field`, `grad_clip_split` [motion_block], `probe` on; the radar memory with the Doppler static gate was in the small-set runs and is OFF here, card section 6)"
s = s.replace(a, b)
a = "| `t14_dc_ctl_s0.json`, `t14_dc_fusion_s0.json` |"
assert s.count(a) == 1
row = "| `t16_ctl.json` | experiment t16_full_network: the matched control of `t16_full.json` (`t10_dual_long_s0`'s recipe without any of the new blocks, probe on); `python -m trainsuite start ../bev_fusion/training/configs/t16_ctl.json --id t16.ctl_s0 ...` | written 2026-11-02 |\n"
s = s.replace(a, row + a)
open("bev_fusion/temp/t16/configs_README2.md", "w", encoding="utf-8").write(s)
print("ok")
