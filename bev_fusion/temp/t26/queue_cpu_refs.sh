#!/bin/bash
# queue the three CPU jobs of the base side through run.py (hard rule 3: runs that feed a verdict), chained with --after; from the bev_fusion folder
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=../bev_fusion
R="python3 ../autoresearch/tools/run.py --seed 0 --retries 0 --require-card --split val --wait-after 1200"
setsid nohup $R --id t26.base_eval --hyp "The stored base runs/t20_base_s0 (GT v9, frozen cached teachers), scored on GT v13 by the live evaluator on VAL 1000, is the control of t26; the GT shift is measured before the launch." --pred "detf mAP within 0.02 of the stored card on GT v9 (0.4092) and lane divider IoU@0.5 within 0.02 (0.4256)" --rule "no verdict of its own: the numbers are the control of t26 (JSON clauses against base); the shift against the stored card is written into the card" -- bash temp/t26/job_base_eval.sh > temp/t26/q1.log 2>&1 < /dev/null &
sleep 20
setsid nohup $R --id t26.base_eval43 --after t26.base_eval --hyp "The same base on all val frames of the 43 scenes where GaussianCaR and CRN both exist (1620 frames) is the control for the SOA object row." --pred "detf mAP within 0.05 of the VAL 1000 value (a 43-scene subset)" --rule "no verdict of its own: input of the by-hand SOA verdict composed from the table rows" -- bash temp/t26/job_base_eval43.sh > temp/t26/q2.log 2>&1 < /dev/null &
sleep 20
setsid nohup $R --id t26.ref_maps_base --after t26.base_eval43 --hyp "The base is behind GaussianCaR on the map rows inside the claim region (forward below 49.5 m, 43 scenes)." --pred "d1 below -0.10 on crossing, walkway, car park and lane divider; drivable and road divider within -0.06 .. 0" --rule "by hand: composed from the table rows (card t26 section 1: BEYOND / LEVEL / BEHIND / NOT SEPARATED per row with the per-row margin)" -- bash temp/t26/job_ref_maps_base.sh > temp/t26/q3.log 2>&1 < /dev/null &
sleep 5
echo queued
