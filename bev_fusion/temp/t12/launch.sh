#!/bin/bash
# t12 launches (card experiments/t12_motion_block/experiment.md)
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/gcar/bin/python
HYP="motion block (two camera BEVs + radar Doppler, 3-level pyramid) trained alone on frozen t10_dual_s0 beats the base velocity head (card t12_motion_block)"
PRED="moving-object EPE base ~v=0 level -> block <= 2.0 m/s; box centres <= t11_vel_hurdle_acc2 + 0.1; total_main not worse than 0.5 %"
RULE="KEEP (joint-training card) if block moving EPE < base - band on both seeds and total_main within 0.5 %; band max(0.1 m/s, 2|s0-s1|); inside band TIE (base cheaper), worse CUT; kill if warp check fails"
export AR_PROJECT=../bev_fusion
nohup python3 ../autoresearch/tools/run.py --id t12.mb_s0 --gpu --hyp "$HYP" --pred "$PRED" --rule "$RULE" --result runs/t12_mb_s0/metrics.json -- \
  $PY training/train_motion_block.py --base runs/t10_dual_s0 --ckpt ckpt_00003000.pt --out runs/t12_mb_s0 --seed 0 --steps 3000 --val 500 --workers 4 \
  > runs/t12_mb_s0.launcher.log 2>&1 &
sleep 5
nohup python3 ../autoresearch/tools/run.py --id t12.mb_s1 --gpu --after t12.mb_s0 --hyp "seed twin of t12.mb_s0: $HYP" --pred "$PRED" --rule "$RULE" --result runs/t12_mb_s1/metrics.json -- \
  $PY training/train_motion_block.py --base runs/t10_dual_s0 --ckpt ckpt_00003000.pt --out runs/t12_mb_s1 --seed 1 --steps 3000 --val 500 --workers 4 \
  > runs/t12_mb_s1.launcher.log 2>&1 &
sleep 5
nohup python3 ../autoresearch/tools/run.py --id t12.hurdle_ctl --gpu --hyp "control of t12: the box velocity of t11_vel_hurdle_acc2 at the GT centres of moving objects, same script and frames (--steps 0)" --pred "centre moving EPE of its object head clearly below v = 0 (it revived the velocity: AVE moving 2.86 on VAL 1000)" --rule "read only as the strong baseline of t12 (centre comparison); no verdict of its own" --result runs/t12_hurdle_ctl/metrics.json -- \
  $PY training/train_motion_block.py --base runs/t11_vel_hurdle_acc2 --ckpt ckpt_00001000.pt --out runs/t12_hurdle_ctl --seed 0 --steps 0 --val 500 --workers 2 \
  > runs/t12_hurdle_ctl.launcher.log 2>&1 &
sleep 2
echo launched
