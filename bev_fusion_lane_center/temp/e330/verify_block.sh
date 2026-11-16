#!/bin/bash
# for each candidate: who names it in CODE/SHELL outside the candidate set itself (scratch temp/ and records excluded)
cd "$(dirname "$0")/../.." || exit 1
CAND="training/search/p16_triage.py training/models/p8_assign.py training/models/p8_seg.py training/data/p16_chain_data.py training/data/p14_chain_data.py training/models/p13_jhead.py training/models/p13_phead.py training/models/p13_junction.py evaluation/loops/loop12/p12_calib.py evaluation/loops/loop17/p17_consumer.py training/train/p17_mapft.py training/train/p18_regis_train.py evaluation/drivers/p18_regis_eval.py evaluation/loops/loop17/p17_series.py training/models/p16_interface.py tools/catalogue/e2e_basis.py tools/catalogue/build_experiment_table.py tools/catalogue/e2e_catalogue.py tools/maintenance/make_structure.py tools/viz/assign_view.py evaluation/chain_eval.py"
for f in $CAND; do
  s=$(basename "$f" .py)
  echo "== $f"
  grep -rIlw --include=*.py --include=*.sh --include=*.json --include=*.csv --exclude-dir=__pycache__ --exclude-dir=temp --exclude-dir=_to_delete --exclude-dir=runs --exclude-dir=results --exclude-dir=experiment_execution --exclude-dir=literature "$s" . ../autoresearch ../GtGenerator ../bev_fusion 2>/dev/null | grep -v "/$s.py$" | while read q; do
    keep=1
    for c in $CAND; do [ "./$c" = "$q" ] && keep=0; done
    [ $keep = 1 ] && echo "   OUTSIDE: $q"
  done
done
echo DONE
