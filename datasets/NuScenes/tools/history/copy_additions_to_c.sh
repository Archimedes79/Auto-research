#!/usr/bin/env bash
# One-off (2026-10-25): derived nuScenes data of the lane project (CRN info files, filtered radar; full set and mini)
# goes into the project folder on C: (bev_fusion_lane_center/data/nuscenes_additions/). Streams WSL tar into
# Windows tar.exe and writes manifests for verify_raw_copy.sh. Run from the repository root.
set -euo pipefail
dst_rel=bev_fusion_lane_center/data/nuscenes_additions
mkdir -p "$dst_rel/lcr_crn" "$dst_rel/lcr_crn_mini" "$dst_rel/manifests"
dst="$(cd "$dst_rel" && pwd)"
copy() {  # copy <source dir> <target dir> <name> <members...>
  local src="$1" to="$2" name="$3"; shift 3
  ( cd "$src" && find -L "$@" -type f -printf '%p\t%s\n' | sort ) > "$dst/manifests/$name.tsv"
  echo "$(date +%T) $name: $(wc -l < "$dst/manifests/$name.tsv") files"
  tar -C "$src" -chf - "$@" | tar.exe -xf - -C "$(wslpath -w "$to")"
  echo "$(date +%T) $name copied"
}
add="nuscenes_infos_train.pkl nuscenes_infos_val.pkl nuscenes_infos_train_crn.pkl nuscenes_infos_val_crn.pkl radar_bev_filter radar_pv_filter"
# shellcheck disable=SC2086
copy "$HOME/data/nuscenes_additions/lcr_crn" "$dst/lcr_crn" lcr_crn $add
# shellcheck disable=SC2086
copy "$HOME/data/nuscenes" "$dst/lcr_crn_mini" lcr_crn_mini $add
echo "$(date +%T) all copied"
