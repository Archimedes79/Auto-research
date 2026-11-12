#!/bin/bash
cd "$(dirname "$0")/../.."
mkdir -p runs/_superseded
for k in 3 4; do
  if [ -d runs/e273_p_s0_p$k ]; then mv runs/e273_p_s0_p$k runs/_superseded/e273_p_s0_v2_p$k; fi
done
ls runs/_superseded; ls -d runs/e273_p_s0* 
