#!/usr/bin/env bash
# Restart of the study lane after the machine/WSL went down on 2026-10-28 08:41 (camtemp_none resumes from ckpt 2000).
cd "$(dirname "$0")/../.." || exit 1
sleep 36000 &                       # keeps the WSL VM alive
bash temp/study/move_ckpts.sh --loop > temp/study/move_ckpts.log 2>&1 &
exec bash temp/study/lane.sh 0a3d73c camtemp_none fusion_gate radar_pillars trunk_second trunk_small lift_s camtemp_gru fusion_cmx ref_s1
