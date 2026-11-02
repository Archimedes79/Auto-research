#!/usr/bin/env bash
# Build the WSL "views" of nuScenes that existing code reads (~/data/nuscenes_full, ~/data/nuscenes): folders of
# symlinks only - raw entries point to the raw data on D: (D:\Auto-research\NuScenes), the lane project's additions
# point to its project folder on C: (bev_fusion_lane_center/data/nuscenes_additions). The former real folders in
# WSL are moved to ~/_to_delete/ (purge only with Mirko's OK). Re-run after the repository folder or the D: root
# moves (the links are absolute). Refuses while a process has a file open under ~/data (checked on the link TARGETS: an
# open file reached through a view shows as /mnt/d/... or as the additions path in /proc/*/fd, never as ~/data/...).
# Usage (from the repository root): wsl bash data/NuScenes/tools/link_views.sh [--selftest]
#   --selftest: holds a file of the mini view open and checks that the guard sees it; changes nothing
set -euo pipefail
raw_d=/mnt/d/Auto-research/NuScenes
add_c="$(cd bev_fusion_lane_center/data/nuscenes_additions && pwd)"
add="nuscenes_infos_train.pkl nuscenes_infos_val.pkl nuscenes_infos_train_crn.pkl nuscenes_infos_val_crn.pkl radar_bev_filter radar_pv_filter"
open_files() {  # "fd -> target" of every open file under a view or under what the views link to (no pipe: pipefail + SIGPIPE)
  find /proc/[0-9]*/fd \( -lname "$HOME/data/*" -o -lname "$raw_d/nuscenes_full/*" -o -lname "$raw_d/nuscenes_mini/*" \
    -o -lname "$add_c/*" \) -printf '%p -> %l\n' 2>/dev/null || true
}
if [ "${1:-}" = "--selftest" ]; then
  f="$HOME/data/nuscenes/LICENSE"
  exec 3< "$f"                                       # this shell holds a file of the mini view open
  hit=$(open_files)
  exec 3<&-
  case "$hit" in
    *"/proc/$$/fd/3 -> $raw_d/nuscenes_mini/LICENSE"*) echo "selftest ok: the guard sees $f open (as $raw_d/nuscenes_mini/LICENSE)" ;;
    *) echo "selftest FAILED: the guard does not see $f open"; exit 1 ;;
  esac
  exit 0
fi
open=$(open_files)
if [ -n "$open" ]; then echo "ABORT: files under the views (~/data) are open:"; echo "$open" | head -n 5; exit 1; fi
stamp=$(date +%Y-%m-%d_%H%M%S)
mkdir -p "$HOME/_to_delete"
view() {  # view <view path> <raw dir on D:> <additions dir on C:> <raw entries...>
  local v="$1" r="$2" a="$3"; shift 3
  local new="$v.view_new"
  rm -rf "$new"; mkdir -p "$new"
  for e in "$@"; do [ -e "$r/$e" ] || { echo "ABORT: $r/$e missing"; exit 1; }; ln -s "$r/$e" "$new/$e"; done
  for e in $add; do [ -e "$a/$e" ] || { echo "ABORT: $a/$e missing"; exit 1; }; ln -s "$a/$e" "$new/$e"; done
  if [ -e "$v" ] && [ ! -L "$v" ]; then mv "$v" "$HOME/_to_delete/$(basename "$v")_wsl_copy_$stamp"; fi
  mv "$new" "$v"
  echo "view $v:"; ls -la "$v" | tail -n +4
}
cd "$raw_d/nuscenes_full"; full_entries=$(ls -A)
cd "$raw_d/nuscenes_mini"; mini_entries=$(ls -A)
# shellcheck disable=SC2086
view "$HOME/data/nuscenes_full" "$raw_d/nuscenes_full" "$add_c/lcr_crn" $full_entries
# shellcheck disable=SC2086
view "$HOME/data/nuscenes" "$raw_d/nuscenes_mini" "$add_c/lcr_crn_mini" $mini_entries
for old in "$HOME/data/nuscenes_additions" "$HOME/downloads/nuscenes_radar_456" "$HOME/downloads/nuscenes_camera"; do
  [ -e "$old" ] && mv "$old" "$HOME/_to_delete/$(basename "$old")_$stamp"
done
du -sh "$HOME/_to_delete"/*_"$stamp" 2>/dev/null || true
