#!/bin/bash
# E389 (D128): the read-out of ONE trained arm, CPU, run by run.py as E389.read_<tag> (temp/e389/e389_read.sh launches them).
# usage: bash temp/e389/e389_read_one.sh <tag> <cache seed> <reference tag>      tag = m_s0 | z_s0 | z2_s0 | ...      the reference (z of the same seed) is read in mode error
# arm m: dump in the five map modes, then the paired scores against the reference z (mode error: the matched no-map head; arm z ignores the map, so its one dump serves every mode);
# arm z / z2 / zg: the dump in mode error and the single score (z2 and zg also paired against z of seed 0 / the same seed: the seed-twin noise).
# Never edit this file while a run reads it.
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0 OMP_NUM_THREADS=2
TAG=$1; S=$2; REF=$3
EV="bash tools/env/_py14.sh jroute_eval.py"
case "$TAG" in
  m_*|m2_*|mg_*)
    for MODE in error exact none wrong gross; do
      $EV dump --run e389_$TAG --seed $S --mode $MODE --cpu || exit 1
    done
    for MODE in error exact none wrong gross; do
      $EV score --run e389_$TAG --seed $S --mode $MODE --ref-run e389_$REF --ref-mode error || exit 1
    done
    $EV score --run e389_$TAG --seed $S --mode wrong --ref-run e389_$TAG --ref-mode error || exit 1     # the kill clause: the head against itself, wrong map vs the right one (diff within 0.01 = it ignores the map)
    ;;
  *)
    $EV dump --run e389_$TAG --seed $S --mode error --cpu || exit 1
    $EV score --run e389_$TAG --seed $S --mode error || exit 1
    if [ "$TAG" != "$REF" ]; then
      $EV score --run e389_$TAG --seed $S --mode error --ref-run e389_$REF --ref-mode error || exit 1
    fi
    ;;
esac
echo "read $TAG ok"
