#!/bin/bash
cd "$(dirname "$0")/../.."
declare -A R
R[j0_s0]="kp_f1 0.580 (c 0.400, +0.180); best-thr F1 0.617; zone IoU 0.721; core IoU 0.483"
R[j0_s1]="kp_f1 0.554 (c 0.377, +0.177); best-thr F1 0.606; zone IoU 0.718; core IoU 0.475"
R[j1_s0]="kp_f1 0.630 (c 0.400, +0.231); best-thr F1 0.653; zone IoU 0.760; core IoU 0.571"
R[j1_s1]="kp_f1 0.642 (c 0.377, +0.265); best-thr F1 0.666; zone IoU 0.759; core IoU 0.578"
R[j2_s0]="kp_f1 0.640 (c 0.400, +0.240); best-thr F1 0.662; zone IoU 0.764; core IoU 0.589"
R[j2_s1]="kp_f1 0.636 (c 0.377, +0.259); best-thr F1 0.663; zone IoU 0.762; core IoU 0.583"
for k in j0_s0 j0_s1 j1_s0 j1_s1 j2_s0 j2_s1; do
  python3 ../autoresearch/tools/ledger.py note E286.$k --result "${R[$k]}" --decision "KEEP by hand: keypoint scorer, rule of the card (kp_f1 > c + 0.05 on both seeds); base for E287 = j1 (tie j1 / j2 inside 0.01, cheaper arm)" --notes "scored 2026-11-12 with the fixed jmaps_eval; card E286 section 7"
done
