cd /mnt/c/Auto-research/bev_fusion
S=/mnt/c/Users/mmeut/AppData/Local/Temp/claude/C--Auto-research/6e9638e0-a42f-4660-b7d5-cac910d9084d/scratchpad/night
pkill -f "[b]ev_queue3.sh"; sleep 1; pgrep -fa "[b]ev_queue3" || echo "queue3 stopped"
python3 $S/make_cfg.py training/configs/t11_vel_hurdle.json training/configs/t11_vel_hurdle_acc2.json \
  name='"t11_vel_hurdle_acc2"' accum=2 \
  _about='"arm B2 of t11_vel_hurdle (2026-11-01 01:05, lead): as arm B prime (fresh vel branch + hurdle) with gradient accumulation 2 (effective batch 6): every run with effective batch 3 has a dead velocity output, the alive ones had >= 6 (t11_vel_cause / why-the-box-velocity session, temp/box_vel/out/alive_table.txt)"'
grep -E '"accum"|"name"|"_about"' training/configs/t11_vel_hurdle_acc2.json | cut -c1-120
