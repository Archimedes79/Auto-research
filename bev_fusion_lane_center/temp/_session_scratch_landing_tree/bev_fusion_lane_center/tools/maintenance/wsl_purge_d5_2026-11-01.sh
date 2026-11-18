#!/bin/bash
# D5 purge of the WSL caches nobody reads (PURGE_2026-10-31.md section 2), prepared 2026-11-01. NOTHING here has run.
#   bash wsl_purge_d5_2026-11-01.sh            dry run: lists what would go and its size
#   bash wsl_purge_d5_2026-11-01.sh --delete   removes exactly this list
# fused32 (GaussianCaR fused features, regenerable, read only by the --feats fused32 option), the mini-split copies, review rounds and setup scripts of loops 1 - 9.
shopt -s nullglob
ITEMS=(
  "$HOME/exp_gcar_b123/feats/train_fused32.npy"
  "$HOME/exp_gcar_b123/feats/val_fused32.npy"
  "$HOME/exp_crn_mini"
  "$HOME/exp_gcar_mini"
  "$HOME/_to_delete"
  "$HOME"/_review_*
  "$HOME"/_verify_*
  "$HOME/extract_blob.sh" "$HOME/run_full_stage.sh" "$HOME/run_py.sh" "$HOME/setup_env.sh" "$HOME/trim_sweeps.sh"
)
total=0; n=0
for p in "${ITEMS[@]}"; do
  [ -e "$p" ] || continue
  s=$(du -sk "$p" | cut -f1); total=$((total + s)); n=$((n + 1))
  if [ "$1" = "--delete" ]; then rm -rfv -- "$p" | tail -1; else echo "would remove $p ($((s / 1024)) MB)"; fi
done
echo "$n items, $((total / 1048576)) GB $( [ "$1" = "--delete" ] && echo removed || echo 'would go (dry run)')"
