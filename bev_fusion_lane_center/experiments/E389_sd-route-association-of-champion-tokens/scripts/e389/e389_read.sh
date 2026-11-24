#!/bin/bash
# E389 (D128): the read-outs of the trained arms through run.py (CPU, the card's own words, ids E389.read_<tag>): z_s0 + z_s1 first (the references), then z2_s0 + m_s0 + m_s1 + m2_s0 (+ zg_s0 / mg_s0 when G=1).
# usage: [G=0] bash temp/e389/e389_read.sh      log: temp/e389/chain.log, per read temp/e389/read_<tag>.out.   Never edit this file while it runs.
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
eval "$(python3 temp/e389/e389_texts.py)"
L=temp/e389/chain.log
say() { echo "$(date '+%m-%d %H:%M:%S') [e389 read] $*" >> $L; }
RUN="python3 ../autoresearch/tools/run.py"
rd() {   # rd <tag> <seed> <reference tag>
  say "read $1"
  $RUN --id E389.read_$1 --seed $2 --ram-mb 4000 --retries 1 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash temp/e389/e389_read_one.sh $1 $2 $3 >> temp/e389/read_$1.out 2>&1
  say "read $1 ended (exit $?)"
}
rd z_s0 0 z_s0 &
rd z_s1 1 z_s1 &
wait
rd z2_s0 0 z_s0 &
rd m_s0 0 z_s0 &
rd m_s1 1 z_s1 &
rd m2_s0 0 z_s0 &
wait
if [ "${G:-0}" = "1" ]; then
  rd zg_s0 0 z_s0 &
  wait
  rd mg_s0 0 zg_s0
fi
say "all reads ended"
