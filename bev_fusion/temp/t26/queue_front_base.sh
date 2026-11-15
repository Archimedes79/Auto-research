#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=../bev_fusion
python3 ../autoresearch/tools/run.py --seed 0 --retries 0 --require-card --split val --id t26.ref_front_base --hyp "CRN (reference) against the base on the front sector (0 - 50 m, +-32.3 degrees) of the 1620 common frames: the base is behind CRN on vehicles and pedestrians." --pred "mAP2 of CRN above the base's by more than 0.10 (CRN devkit mAP 0.50 on these samples)" --rule "no verdict of its own: the object row of the by-hand SOA verdict composed from the table rows (card t26 section 1)" -- bash temp/t26/job_front_base.sh
echo "rc=$?"
tail -n 40 records/runs/t26.ref_front_base/run.log | cut -c1-240
