#!/usr/bin/env bash
# ledger note of pitch.maps_lfl_check
cd "$(dirname "$0")/../.." || exit 1
export AR_PROJECT=../bev_fusion
python3 ../autoresearch/tools/ledger.py note pitch.maps_lfl_check \
  --result 'GPU re-score of t1_r4_soa (400 frames) against the stored CPU run: max abs difference over every number (layers, distance bins, thresholds, tolerances) 0.00018 for ours, 0.0 for GaussianCaR; headline numbers differ by at most 5e-05 (drivable 0.7188 / 0.7188, lane divider 0.2939 / 0.2938)' \
  --decision 'prediction hit (within 0.01, GaussianCaR block identical): the GPU numbers of pitch.maps_lfl stand'
python3 ../autoresearch/tools/ledger.py tail 2>&1 | grep -E "pitch.maps" | cut -c1-160
