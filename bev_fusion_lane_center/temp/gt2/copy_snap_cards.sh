cd "$(dirname "$0")/../.."
# the confirmation scores of E311 were written by a scratch scorer into temp/gt2/snap/<run>_thr<t>.json; they are the same numbers as `jends_eval.py score --snap --thr t` (same scorer, same code path): copy them to the official names in the run folders (never over an existing file)
for f in temp/gt2/snap/*.json; do
  b=$(basename $f .json); run=${b%_thr*}; thr=${b##*_thr}
  d=runs/$run/ends_card_snap_thr$thr.json
  [ -f $d ] || cp $f $d
done
ls runs/e287_p2_s0/ | grep card; ls runs/e305_e_s1/ | grep card
