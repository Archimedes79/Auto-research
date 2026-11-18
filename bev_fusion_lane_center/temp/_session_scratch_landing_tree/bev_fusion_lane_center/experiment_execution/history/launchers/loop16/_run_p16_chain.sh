#!/bin/bash
# Loop 16 rung 3: navigation chain on the champion builder.  Usage: _run_p16_chain.sh <stage A|B> <tag> <design> <run name> <tau> [assembly keywords ...]
#  stage A (no TEST): tokens of train + val_clean, route head seed 0 and seed 1 (E104b recipe: 40000 iterations, xattn), route evaluation on VAL (cells exact, s1, s2, s3)
#  stage B (TEST, once): tokens of test_clean, route evaluation on TEST (exact s1 s2 s3 wrong none), chain data, consumer (route + ego, 3 seeds)
ST=$1; TAG=$2; DES=$3; NAME=$4; TAU=$5; shift 5
ASM="$@"
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/p16chain_$TAG
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
mkdir -p $O
P="bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh"
if [ "$ST" = "A" ]; then
  rm -f $O/A.done
  for SP in val_clean train; do
    [ -f ~/exp_gcar_b123/p10/tok_p16_${TAG}_${SP}.pkl ] || $P p16_tokens.py --design $DES --name $NAME --out tok_p16_$TAG --split $SP --tau $TAU ${TOKSET:+--set $TOKSET} --asm $ASM > $O/tokens_$SP.log 2>&1
  done
  $P p10_route_train.py --tok tok_p16_${TAG}_train --name p16_${TAG}_route_s0 --iters 40000 --bs 8 --lr 0.0007 --workers 3 --xattn --seed 0 > $O/route_s0.log 2>&1
  $P p10_route_train.py --tok tok_p16_${TAG}_train --name p16_${TAG}_route_s1 --iters 40000 --bs 8 --lr 0.0007 --workers 3 --xattn --seed 1 > $O/route_s1.log 2>&1
  for SD in s0 s1; do
    $P p10_route_eval2.py --tok tok_p16_${TAG}_val_clean --split val_clean --model p16_${TAG}_route_$SD --kind x --taus 0.5 --cells exact,s1,s2,s3 --tag val > $O/eval_val_$SD.log 2>&1
  done
  echo done > $O/A.done
else
  rm -f $O/B.done
  [ -f ~/exp_gcar_b123/p10/tok_p16_${TAG}_test_clean.pkl ] || $P p16_tokens.py --design $DES --name $NAME --out tok_p16_$TAG --split test_clean --tau $TAU ${TOKSET:+--set $TOKSET} --asm $ASM > $O/tokens_test.log 2>&1
  for SD in s0 s1; do
    $P p10_route_eval2.py --tok tok_p16_${TAG}_test_clean --split test_clean --model p16_${TAG}_route_$SD --kind x --taus 0.5 --cells exact,s1,s2,s3,wrong,none --tag test > $O/eval_test_$SD.log 2>&1
  done
  $P p16_chain_data.py --tok tok_p16_$TAG --route p16_${TAG}_route_s0 --out p16chain_$TAG --draws 3 --workers 4 > $O/chain_data.log 2>&1
  $P p14_chain_consumer.py --data p16chain_$TAG --out p16_cons_$TAG --modes route+ego --seeds 3 > $O/consumer.log 2>&1
  echo done > $O/B.done
fi
