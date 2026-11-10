#!/usr/bin/env bash
# One-off (t2_block_swaps + t3 tries): GPU job queue, one job at a time, the queue file is re-read before every job so it can be edited
# while the lane runs (first line = next job). Jobs:  train:<arm>   train t2_<arm> (resumes), then evaluate 1000 VAL keyframes
#                                                     ctl:<arm>:<control>   sanity control of a finished arm (checkpoint from D:)
#                                                     cfg:<name>   train training/configs/<name>.json (name = run name), then evaluate
# Every job is idempotent (skipped when its result exists). Usage: bash temp/study/lane2.sh   (log: temp/study/lane2.log)
cd "$(dirname "$0")/../../.." || exit 1
Q=temp/study/queue.txt
PY=~/miniconda3/envs/gcar/bin/python
REV=0a3d73c
sleep 36000 &                                         # keeps the WSL VM alive
bash temp/study/move_ckpts.sh --loop > temp/study/move_ckpts.log 2>&1 &
evalrun() {  # <run dir> [extra args]
  r=$1; shift
  $PY -u evaluation/bev_suite.py $r --workers 4 "$@" eval --n 1000 > temp/study/eval_$(basename $r)$LOGSFX.log 2>&1
  tail -n 1 temp/study/eval_$(basename $r)$LOGSFX.log
}
while true; do
  job=$(grep -v '^\s*$' $Q | head -n 1)
  [ -z "$job" ] && { echo "queue empty $(date +%T)"; sleep 60; job=$(grep -v '^\s*$' $Q | head -n 1); [ -z "$job" ] && break; }
  grep -v -x -F "$job" $Q > $Q.tmp; mv $Q.tmp $Q
  IFS=: read kind a c <<< "$job"
  LOGSFX=""
  case $kind in
    train|cfg)
      if [ $kind = train ]; then name=t2_$a; cfgname=t2_$a; else name=$a; cfgname=$a; fi
      r=runs/$name
      if [ -f $r/eval/metrics.json ]; then echo "$job already evaluated, skip"; continue; fi
      echo "== $job train $(date +%T)"
      [ -f $r/.done ] || { cp $r/train.log $r/train.log.part$(date +%s) 2>/dev/null; bash studies/recipes/train/run_cfg.sh training/configs/$cfgname.json $REV; }
      [ -f $r/.done ] || { echo "$job: no .done, skipped evaluation"; continue; }
      echo "== $job eval $(date +%T)"
      evalrun $r ;;
    ctl)
      r=runs/t2_$a
      [ -f $r/eval/metrics_ctl_$c.json ] && { echo "$job done already"; continue; }
      echo "== $job $(date +%T)"
      LOGSFX=_$c
      d=""; ls $r/ckpt_*.pt > /dev/null 2>&1 || d="--ckpt-dir /mnt/d/Auto-research/bev_fusion_ckpts/t2_$a"
      evalrun $r --control $c $d ;;
  esac
done
echo "lane finished $(date +%T)"
