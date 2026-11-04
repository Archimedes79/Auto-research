#!/bin/bash
# scratch: the bundle arm t18.fix behind the data arm's evaluation, with a 20-step smoke of its flags in front (a failed smoke refuses the 6-hour run), then its evaluation
cd "$(dirname "$0")/../.." || exit 1               # bev_fusion/
export AR_PROJECT=/mnt/c/Auto-research/bev_fusion
export AR_COMMIT=5e8513ee07f6c55a9dc26474b73a815babb6583a   # the code of main when the Training chat's hygiene flags and lr_groups were in (df9d437 + docs): pinned, whatever is committed later
RUN=/mnt/c/Auto-research/autoresearch/tools/run.py
cd ../autoresearch
python3 -m trainsuite start ../bev_fusion/training/configs/t18_fix.json --run-arg=--no-git --id t18.smoke --preset smoke --preset small_set --no-card --go --run-arg=--after --run-arg=t17.data.eval 2>&1 | grep -E 'check:|started|refus|error' | cut -c1-200
sleep 3
HYP="The bundle of the module fixes (the Training chat's hygiene flags: weights w_unknown / w_interpolated applied, unknown velocity out of the moving / static BCE and the velocity losses, dead radar_slot branch removed) and the motion block on its own learning rate (1e-4, warm-up 1000), on the 700-scene recipe of t17.data, improves the static-cell velocity error of the motion block without losing the moving cells and keeps the maps and objects (t18 card)."
PRED="Against t17.data (single seed each, VAL 1000 frames, GT v8, written before either number exists): static EPE (vel/dt+0.00/cells/static/epe) at most 0.6 x the control's and at most 0.30 m/s; moving vehicle EPE and box-centre EPE not worse than the control by more than their band; maps and mAP2 inside their bands."
RULE="By hand: one seed per arm. ADOPT as the delivered system / M default: static EPE below the control by more than 3 x its band AND moving / centre EPE not worse by more than the band AND the guards hold; static better by 1 - 3 bands: CANDIDATE (seed twin); inside the band: TIE (keep t17.data); moving or centre worse by more than the band: CUT. A flag that touches M's optimisation makes the label 'bundle: lr + flag'. Kill: non-finite loss in more than 1 percent of the steps; motion@+0.0 at step 2000 more than 50 percent above t17.data's."
python3 -m trainsuite start ../bev_fusion/training/configs/t18_fix.json --run-arg=--no-git --id t18.fix --hyp "$HYP" --pred "$PRED" --rule "$RULE" --go --run-arg=--after --run-arg=t18.smoke 2>&1 | grep -E 'check:|started|refus|error' | cut -c1-200
cd ../bev_fusion
sleep 3
nohup python3 $RUN --id t18.fix.eval --gpu --split val --no-card --after t18.fix -- bash temp/t16/eval_chain.sh runs/t18_fix > runs/t18_fix.eval.launcher.log 2>&1 &
sleep 3
ps -eo pid,etimes,cmd | grep 'run.py --id t1[678]' | grep -v grep | cut -c1-110
