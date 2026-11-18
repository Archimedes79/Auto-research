#!/bin/bash
# E291 rule evaluation for the given seeds (pooled paired scene bootstrap, twin = control): usage e291_rule.sh 0   or   e291_rule.sh 0 1
# waits for the cards of every seed given, then runs the G / G-unsnapped / R / C / kill(5) s / kill(5) sc commands of temp/e291/install_plan.md section 3 and the two analysis commands,
# each as a run.py job (hard rule 3); results in experiment_execution/runs/E291.rule_<name>_<seeds>/run.log and in temp/_session_scratch_s292/logs/e291_rule_<seeds>.log
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
SEEDS="$*"; TAG=$(echo $SEEDS | tr -d ' ')
LOG=temp/_session_scratch_s292/logs/e291_rule_$TAG.log
say() { echo "$* $(date +%H:%M:%S)" >> $LOG; }
for S in $SEEDS; do
  for f in runs/e291_b_s${S}_clean/ends_card_clean_snap_thr0.1.json runs/e291_b_s${S}_noise30/ends_card_noise30_snap_thr0.1.json runs/e291_b_s${S}_u/ends_card_u_snap_thr0.1.json runs/e291_b_s${S}_s/ends_card_s_snap_thr0.1.json runs/e291_b_s${S}_sc/ends_card_sc_snap_thr0.1.json runs/e291_b_s${S}_clean/ends_card_clean_thr0.1.json runs/e291_c_s${S}/ends_card_snap_thr0.1.json runs/e291_c_s${S}/ends_card_thr0.1.json; do
    n=0; until [ -s $f ]; do n=$((n+30)); [ $n -gt 28800 ] && { say "missing $f after 8 h"; exit 2; }; sleep 30; done
  done
done
say "cards of seeds $SEEDS present"
rp() { # id, then the command after --
  local id=$1; shift
  python3 ../autoresearch/tools/run.py --id $id --seed 0 --retries 0 --require-card --split val --timeout 60 \
    --hyp "E291 rule evaluation (card section 0): $id" --pred "G / R / C / kill (5) as the card's three-valued rule; likely INCONCLUSIVE (underpowered)" \
    --rule "E291 card section 0 (by hand: replay, decide.py is not used): pooled paired scene bootstrap of the mean over seeds of the snapped topo F1 difference against the recipe twin; the clean row is an upper bound" -- "$@" > temp/_session_scratch_s292/logs/${id}.out 2>&1
  say "$id exit $?"
}
args_pair() { # a_run a_card b_run b_card for all seeds
  local ra=$1 ca=$2 rb=$3 cb=$4 out=""
  for S in $SEEDS; do out="$out ${ra/SS/$S} $ca ${rb/SS/$S} $cb"; done
  echo $out
}
J="bash tools/env/_py14.sh jdiag.py boot_pooled"
rp E291.rule_G_$TAG $J $(args_pair e291_b_sSS_clean ends_card_clean_snap_thr0.1 e291_c_sSS ends_card_snap_thr0.1)
rp E291.rule_Gunsnap_$TAG $J $(args_pair e291_b_sSS_clean ends_card_clean_thr0.1 e291_c_sSS ends_card_thr0.1)
rp E291.rule_R_$TAG $J $(args_pair e291_b_sSS_noise30 ends_card_noise30_snap_thr0.1 e291_c_sSS ends_card_snap_thr0.1)
rp E291.rule_C_$TAG $J $(args_pair e291_b_sSS_clean ends_card_clean_snap_thr0.1 e291_b_sSS_u ends_card_u_snap_thr0.1)
rp E291.rule_Ks_$TAG $J $(args_pair e291_b_sSS_s ends_card_s_snap_thr0.1 e291_c_sSS ends_card_snap_thr0.1)
rp E291.rule_Ksc_$TAG $J $(args_pair e291_b_sSS_sc ends_card_sc_snap_thr0.1 e291_c_sSS ends_card_snap_thr0.1)
rp E291.rule_splits_$TAG bash tools/env/_py14.sh analysis.py splits $(args_pair e291_b_sSS_clean ends_card_clean_snap_thr0.1 e291_c_sSS ends_card_snap_thr0.1)
for S in $SEEDS; do rp E291.rule_gate_s${S} bash tools/env/_py14.sh analysis.py gate e291_b_s${S}_noise30 e291_b_s${S}_clean; done
say "e291 rule $TAG done"
