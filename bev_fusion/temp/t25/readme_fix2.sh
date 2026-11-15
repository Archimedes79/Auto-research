set -e
f=tests/README.md
sed -i '55s|RESULT_PLACEHOLDER|about 190 s under load): 9 / 9 killed (radar carry ignored, radar frames in the wrong order, masks in the wrong order, radar state not carried, fused state cut by a detach, radar side loss without features / from the oldest frame, wrong sign of v / of omega in the radar ego chains; the last four were found unguarded by the review of 2026-11-14 and are pinned since)|' $f
sed -i '91s|about 2 min|about 3 min|' tests/run_gates.sh
sed -i 's|tiny cached model, CPU, no data, about 2 min\.|tiny cached model, CPU, no data, about 3 min.|' tests/test_radar_recursive.py
