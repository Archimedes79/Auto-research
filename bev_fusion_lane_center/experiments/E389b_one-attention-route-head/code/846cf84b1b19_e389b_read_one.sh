#!/bin/bash
# E389b: the read-out of ONE trained arm, CPU, run by run.py as E389b.read_<tag>.
# usage: bash temp/e389b/e389b_read_one.sh <tag> <cache seed>      tag = a_s0 | a_s1 | p_s0
# the five map modes, then the paired scores against the E389 z of the same seed (mode error), and the arm against the E389 m of the same seed (a - m; only for tags a_*)
# Never edit this file while a run reads it.
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0 OMP_NUM_THREADS=2
TAG=$1; S=$2
EV="bash tools/env/_py14.sh jroute_eval.py"
for MODE in error exact none wrong gross; do
  $EV dump --run e389b_$TAG --seed $S --mode $MODE --cpu || exit 1
done
for MODE in error exact none wrong gross; do
  $EV score --run e389b_$TAG --seed $S --mode $MODE --ref-run e389_z_s$S --ref-mode error || exit 1
done
$EV score --run e389b_$TAG --seed $S --mode wrong --ref-run e389b_$TAG --ref-mode error || exit 1     # the kill clause: the head against itself, wrong map vs the right one
case "$TAG" in
  a_*) $EV score --run e389b_$TAG --seed $S --mode error --ref-run e389_m_s$S --ref-mode error || exit 1 ;;     # a - m with the paired interval
esac
echo "read $TAG ok"
