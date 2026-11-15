#!/bin/bash
# scratch: three planted bugs in a COPY of trainkit for the keep_steps checks (never the shared tree)
set -u
SRC=/mnt/c/Auto-research/autoresearch/trainkit
PY=~/miniconda3/envs/gcar/bin/python
run_mut () {   # name, file, sed expression
  D=/mnt/c/Auto-research/bev_fusion/temp/cleanup/mut_keep/$1
  rm -rf "$D"; mkdir -p "$D"; cp -r "$SRC" "$D/trainkit"; rm -rf "$D/trainkit/__pycache__"
  sed -i "$3" "$D/trainkit/$2"
  if cmp -s "$SRC/$2" "$D/trainkit/$2"; then echo "$1: NOT APPLIED"; return; fi
  ( cd "$D" && OMP_NUM_THREADS=2 nice -n 15 $PY trainkit/selftest.py > out.txt 2>&1 ); rc=$?
  echo "$1: rc=$rc $( [ $rc -ne 0 ] && echo KILLED || echo SURVIVED )"
}
run_mut m1_no_protect checkpoint.py 's/if f not in protected:/if True:/'
run_mut m2_name_format checkpoint.py 's/{int(s):08d}/{int(s):07d}/'
run_mut m3_no_validation loop.py 's/    if bad_keep:  /    if False:  /'
run_mut m4_final_prune_drops loop.py 's/if (cfg.slim_at_end or was_slim) else cfg.keep, keep_steps)/if (cfg.slim_at_end or was_slim) else cfg.keep, ())/'
run_mut m5_stop_prune_drops loop.py '0,/checkpoint.prune_checkpoints(out_dir, cfg.keep, keep_steps)/s//checkpoint.prune_checkpoints(out_dir, cfg.keep)/'
run_mut m6_not_free config.py 's/"keep", "slim_at_end"/"keep", "keep_steps", "slim_at_end"/'
