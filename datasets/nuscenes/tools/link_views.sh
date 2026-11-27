#!/usr/bin/env bash
# The nuScenes views: folders of links that code reads instead of the raw folders.
#   <data root>/nuscenes       links to every entry of the raw trainval folder, plus the four CRN info files (real files, kept as they are)
#   <data root>/nuscenes_mini  the same for the mini folder
#   ~/data/nuscenes_full -> <data root>/nuscenes, ~/data/nuscenes -> <data root>/nuscenes_mini   (the WSL paths older code reads)
# The data root is "data" of the machine file (~/.research/machine.json, or RESEARCH_MACHINE); the raw root is NUSC_ROOT, else the folder
# NuScenes next to the data root. A former WSL view that is a real folder of links is replaced (refused if it holds anything but links).
# Usage (from the repository root): wsl bash datasets/nuscenes/tools/link_views.sh [--check]   (--check: report only, exit 1 on a problem)
set -euo pipefail
machine="${RESEARCH_MACHINE:-$HOME/.research/machine.json}"
data=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["data"])' "$machine")
raw="${NUSC_ROOT:-$(dirname "$data")/NuScenes}"
check=0; [ "${1:-}" = "--check" ] && check=1
bad=0
INFOS="nuscenes_infos_train.pkl nuscenes_infos_val.pkl nuscenes_infos_train_crn.pkl nuscenes_infos_val_crn.pkl"

view() {  # view <view folder> <raw folder> <WSL path>
  local v="$1" r="$2" w="$3" e
  [ -d "$r" ] || { echo "the raw folder $r does not exist: set NUSC_ROOT (datasets/nuscenes/README.md)"; exit 1; }
  [ $check = 1 ] || mkdir -p "$v"
  for e in $(cd "$r" && ls -A); do
    [ -L "$v/$e" ] && [ "$(readlink "$v/$e")" = "$r/$e" ] && continue
    if [ $check = 1 ]; then echo "missing link $v/$e -> $r/$e"; bad=1; else ln -sfn "$r/$e" "$v/$e"; fi
  done
  for e in $INFOS; do
    [ -f "$v/$e" ] || { echo "missing $v/$e (a CRN info file; how it is made: tag v1, bev_fusion_lane_center/data/README.md)"; bad=1; }
  done
  [ -L "$w" ] && [ "$(readlink "$w")" = "$v" ] && return 0
  if [ $check = 1 ]; then echo "$w is not a link to $v"; bad=1; return 0; fi
  if [ -e "$w" ] && [ ! -L "$w" ]; then
    [ -z "$(find "$w" -mindepth 1 -not -type l)" ] || { echo "ABORT: $w holds more than links: move it away by hand"; exit 1; }
    rm -r "$w"                                       # a folder of links only: rm removes the links, never what they point at
  fi
  mkdir -p "$(dirname "$w")"
  ln -sfn "$v" "$w"
  echo "$w -> $v"
}
view "$data/nuscenes" "$raw/nuscenes_full" "$HOME/data/nuscenes_full"
view "$data/nuscenes_mini" "$raw/nuscenes_mini" "$HOME/data/nuscenes"
[ $bad = 0 ] && echo "views ok" || { echo "views: problems above"; exit 1; }
