#!/bin/bash
# D7 steps 5-6 (2026-10-25): ground truth, route and SD map of a frame set, CPU only, new files in ~/exp_gcar_<tag>/ only.
#   bash temp/loop19/run_b456_gt.sh <tag>     (tag b456: the 255 new scenes; gtchk: the reproduction check set)
# Order: graph + poses + exact distance (parallel) -> direction arrays + route -> route heat + SD map. Each stage waits for all its jobs and
# stops when one failed (no job of a failed stage keeps writing while a re-run starts).
set -e
T="$1"
case "$T" in ""|b123|mini) echo "usage: run_b456_gt.sh <tag>, not b123 / mini (their files are made by the original pipelines)"; exit 2 ;; esac
cd "$(dirname "$0")/../.."
E="$HOME/exp_gcar_$T"
[ -s "$E/feats/train_tokens.json" ] && [ -s "$E/feats/val_tokens.json" ] || { echo "no token lists in $E/feats"; exit 2; }
export NUSC_ROOT="$HOME/data/nuscenes_full" NUSC_VERSION=v1.0-trainval GT_FRAME=ego
P="bash tools/env/_py14.sh"
step() { echo "== $(date +%H:%M:%S) $*"; }
waitall() { local rc=0 p; for p in "$@"; do wait "$p" || rc=1; done; return $rc; }
step "graph + poses + exact distance"
$P training/data/prep/make_graph_gt.py "$T" > "$E/log_graph.txt" 2>&1 & a=$!
$P training/data/prep/make_poses.py "$T" > "$E/log_poses.txt" 2>&1 & b=$!
$P training/data/prep/make_dist_gt.py "$E/feats" "$E/feats" --cpu > "$E/log_dist.txt" 2>&1 & c=$!
waitall $a $b $c
tail -n 2 "$E/log_graph.txt" "$E/log_poses.txt" "$E/log_dist.txt"
step "direction arrays + route"
$P training/data/prep/make_gt_arrays.py "$T" > "$E/log_arrays.txt" 2>&1 & a=$!
$P training/data/prep/make_route_gt.py "$T" > "$E/log_route.txt" 2>&1 & b=$!
waitall $a $b
tail -n 2 "$E/log_arrays.txt" "$E/log_route.txt"
step "route heat + SD map"
$P training/data/prep/make_route_heat.py "$T" > "$E/log_heat.txt" 2>&1 & a=$!
$P training/data/prep/sd_precompute.py "$T" > "$E/log_sd.txt" 2>&1 & b=$!
waitall $a $b
tail -n 2 "$E/log_heat.txt" "$E/log_sd.txt"
step "FERTIG $T"
