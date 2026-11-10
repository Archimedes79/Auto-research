#!/bin/bash
# scratch: run safety.py for the old export (old) or the worktree (new), ONE PROCESS PER CONFIG (small RAM peaks on the shared box); usage: run_safety.sh old|new [labels...]
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../.." && pwd)"                # the worktree's bev_fusion folder
MAIN=/mnt/c/Auto-research/bev_fusion
PY=~/miniconda3/envs/gcar/bin/python
which_code=$1; shift
if [ "$which_code" = old ]; then ROOT="$HERE/old/bev_fusion"; else ROOT="$WT"; fi
CFG="$ROOT/training/configs"
ALL="t11_vel_hurdle_acc2=$MAIN/runs/t11_vel_hurdle_acc2/config.json=$MAIN/runs/t11_vel_hurdle_acc2/ckpt_00001000.pt \
t10_dual_s0=$MAIN/runs/t10_dual_s0/config.json=$MAIN/runs/t10_dual_s0/ckpt_00003000.pt \
t10_dual_s1=$MAIN/runs/t10_dual_s1/config.json=$MAIN/runs/t10_dual_s1/ckpt_00003000.pt \
t8_ctl_s0=$MAIN/runs/t8_ctl_s0/config.json=$MAIN/runs/t8_ctl_s0/ckpt_00003000.pt \
t8_ctl_s1=$MAIN/runs/t8_ctl_s1/config.json=$MAIN/runs/t8_ctl_s1/ckpt_00003000.pt \
t16_full.json=$CFG/t16_full.json \
t20_base.json=$CFG/t20_base.json \
t16_full_s0=$MAIN/runs/t16_full_s0/config.json=$MAIN/runs/t16_full_s0/ckpt_00006000.pt \
t20_base_s0=$MAIN/runs/t20_base_s0/config.json=$MAIN/runs/t20_base_s0/ckpt_00006000.pt"
if [ $# -gt 0 ]; then
  SEL=""
  for want in "$@"; do for a in $ALL; do case "$a" in "$want"=*) SEL="$SEL $a";; esac; done; done
else
  SEL="$ALL"
fi
cd "$HERE/cwd" || exit 1                        # cwd with models/pretrained -> the weights (read-only link)
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p "$HERE/res_$which_code"
for spec in $SEL; do
  label=${spec%%=*}
  [ -f "$HERE/res_$which_code/$label.pt" ] && { echo "have $label"; continue; }          # resumable
  $PY "$HERE/safety.py" "$ROOT" "$HERE/res_$which_code/$label.pt" $spec || echo "FAILED $label"
done
