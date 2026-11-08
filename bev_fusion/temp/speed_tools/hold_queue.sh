#!/bin/bash
# Mirko 2026-11-02 ~22:15 (BEV tools chat): "please just do t18.fix ready and then stop all BEV system trainings".
# t18.fix keeps running; the trainings that are still waiting get a HOLD marker (run.py chains stop at it, nothing starts).
cd /mnt/c/Auto-research/autoresearch || exit 1
export AR_PROJECT=../bev_fusion
for id in t16.ctl_s0 speed.d speed.e speed.f; do
  python3 tools/hold.py set "$id" --by "tools chat" --reason "waiting training must not start after t18.fix (Mirko: stop all BEV system trainings)" \
    --words "Mirko, 2026-11-02 in the tools chat: please just do t18.fix ready and then stop all BEV system trainings"
done
python3 tools/hold.py show
