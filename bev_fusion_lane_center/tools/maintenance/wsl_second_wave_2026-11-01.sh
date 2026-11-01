#!/bin/bash
# Second wave of the WSL cache purge (lane project), 2026-11-01. NOTHING here has run. Explicit list of regenerable cache files of old loops
# (p8 segment variants of system A, p10 tokens of ladder designs, p14 chains of old loops); the files the live chain reads are not on it.
#   bash wsl_second_wave_2026-11-01.sh            dry run: prints what would go and the total
#   bash wsl_second_wave_2026-11-01.sh --delete   removes exactly these files
# Run it when no job reads the caches. The WSL image shrinks only after compaction (PURGE_2026-10-31.md section 4).
cd ~/exp_gcar_b123 || exit 1
FILES=(
p8/ds_dev_tmpl_segL.pkl
p8/ds_devq_tmpl_segP.pkl
p8/ds_devq_tmpl_segP1.pkl
p8/ds_devq_tmpl_segP2.pkl
p8/ds_devq_tmpl_segP3.pkl
p8/ds_devq_tmpl_segS2.pkl
p8/ds_test_icp_seg.pkl
p8/ds_test_none_seg.pkl
p8/ds_test_tmpl_segL.pkl
p8/ds_test_tmpl_segP.pkl
p8/ds_test_tmpl_segR.pkl
p8/ds_testq_tmpl_segP.pkl
p8/ds_train_tmpl_segB.pkl
p8/ds_train_tmpl_segC.pkl
p8/ds_train_tmpl_segL.pkl
p10/tok_e120_a_oof0.pkl
p10/tok_e120_a_oof1.pkl
p10/tok_e120_a_oof2.pkl
p10/tok_e120_a_test_clean.pkl
p10/tok_e120_a_val_clean.pkl
p10/tok_e94az_oof0.pkl
p10/tok_e94az_oof1.pkl
p10/tok_e94az_oof2.pkl
p10/tok_e94az_val_clean.pkl
p10/tok_p16_f1m_train.pkl
p10/tok_p16_f1m_val_clean.pkl
p10/tok_p16_f2hz_train.pkl
p10/tok_p16_f2hz_val_clean.pkl
p10/tok_p16_f2l_test_clean.pkl
p10/tok_p16_f2l_train.pkl
p10/tok_p16_f2l_val_clean.pkl
p10/tok_p16_rc1v_train.pkl
p10/tok_p16_rc1v_val_clean.pkl
p10/tok_p16_smoke_test_clean.pkl
p10/tok_p16_smoke_train.pkl
p10/tok_p16_smoke_val_clean.pkl
p14/chain_r2_test.pt
p14/chain_r2_train.pt
p14/chain_r2_val.pt
p14/chain_r_test.pt
p14/chain_r_train.pt
p14/chain_r_val.pt
p14/chainfail_blur_ema_hold_mon_0.5.pt
p14/chainfail_blur_maxdecay_0.95.pt
p14/chainfail_blur_single.pt
p14/chainfail_cam_front3_off_ema_hold_mon_0.5.pt
p14/chainfail_cam_front3_off_maxdecay_0.95.pt
p14/chainfail_cam_front3_off_single.pt
p14/chainfail_dark_ema_hold_mon_0.5.pt
p14/chainfail_dark_maxdecay_0.95.pt
p14/chainfail_dark_single.pt
p14/chainfailval_blur_ema_hold_mon_0.5.pt
p14/chainfailval_blur_maxdecay_0.95.pt
p14/chainfailval_blur_single.pt
p14/chainfailval_cam_front3_off_ema_hold_mon_0.5.pt
p14/chainfailval_cam_front3_off_maxdecay_0.95.pt
p14/chainfailval_cam_front3_off_single.pt
p14/chainfailval_dark_ema_hold_mon_0.5.pt
p14/chainfailval_dark_maxdecay_0.95.pt
p14/chainfailval_dark_single.pt
p14/p16chain_f2h_test.pt
p14/p16chain_f2h_train.pt
p14/p16chain_f2h_val.pt
p14/p16chain_f2l_test.pt
p14/p16chain_f2l_train.pt
p14/p16chain_f2l_val.pt
p14/p16chain_smoke_test.pt
p14/p16chain_smoke_train.pt
p14/p16chain_smoke_val.pt
)
total=0; n=0
for p in "${FILES[@]}"; do
  if [ -f "$p" ]; then
    s=$(stat -c %s "$p"); total=$((total + s)); n=$((n + 1))
    if [ "$1" = "--delete" ]; then rm -v -- "$p"; else echo "would remove $p ($((s / 1048576)) MB)"; fi
  fi
done
echo "$n files, $((total / 1073741824)) GB $( [ "$1" = "--delete" ] && echo removed || echo 'would go (dry run)')"
