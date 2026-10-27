#!/bin/bash
# one-off audit: modality / distance-band diagnostics of both t1 runs (the run names keep "t1_r" in the command line:
# other chats' GPU waiters gate on it)
PY=~/miniconda3/envs/gcar/bin/python
for r in runs/t1_r3_combined runs/t1_r0; do
  $PY temp/audit/diag_modalities.py $r --n 300
done
