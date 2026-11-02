#!/usr/bin/env bash
# Links from the project folders to the derived data on D:. The big derived data of the nuScenes family lies next to the raw data, in
# D:\Auto-research\NuScenes_cache and NuScenes_checkpoints (README "Where everything lives"); a project reaches it through a link in its own data/ folder,
# so code and documents name the project path (data/cache/..., data/checkpoints/...), never a D: path. After a folder on D: is renamed, change the table
# below and run --apply.
# Usage (from the repository root): wsl bash data/NuScenes/tools/link_storage.sh [--check | --apply | --selftest]
#   --check (default)  one line per link: ok / missing / wrong / dangling / real / no target; exit 1 when one is not ok
#   --apply            create or re-point the links that are missing or wrong (ln -sfn); never replaces a real file or folder; refuses to re-point a link
#                      while a process has a file open below its old target
#   --selftest         in a temporary tree: --check finds each defect, --apply repairs what it may and leaves a real folder alone
# Environment (the selftest uses them): LS_REPO (default: the repository), LS_STORE (default /mnt/d/Auto-research), LS_TABLE (the table below)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="${LS_REPO:-$(cd "$here/../../.." && pwd)}"
store="${LS_STORE:-/mnt/d/Auto-research}"
# link (below the repository) | target (below the store)
table="${LS_TABLE:-bev_fusion/data/cache/cam_feats_bevfusion|NuScenes_cache/bev_fusion/cam_feats_bevfusion
bev_fusion/data/cache/cam_feats_parts7_10|NuScenes_cache/bev_fusion/cam_feats_parts7_10
bev_fusion/data/checkpoints|NuScenes_checkpoints/bev_fusion}"

state() {  # state <link> <target> -> ok | missing | wrong | dangling | real | "no target"
  local l="$1" t="$2"
  if [ -L "$l" ]; then
    if [ "$(readlink "$l")" != "$t" ]; then echo wrong
    elif [ -e "$l" ]; then echo ok
    else echo dangling; fi
  elif [ -e "$l" ]; then echo real
  elif [ -e "$t" ]; then echo missing
  else echo "no target"; fi
}
open_below() {  # processes with a file open below a folder (the link target an open file resolves to in /proc/*/fd)
  find /proc/[0-9]*/fd -lname "$1/*" -printf '%p -> %l\n' 2>/dev/null | head -n 3 || true
}
run() {  # run <check|apply>
  local mode="$1" bad=0 l t s old
  while IFS='|' read -r l t; do
    [ -z "$l" ] && continue
    l="$repo/$l"; t="$store/$t"
    s="$(state "$l" "$t")"
    case "$s" in
      ok) echo "ok         $l -> $t" ;;
      missing|wrong)
        if [ "$mode" = apply ]; then
          if [ ! -e "$t" ]; then echo "SKIP       $l: target $t does not exist"; bad=1; continue; fi
          if [ "$s" = wrong ]; then
            old="$(readlink "$l")"; [ -n "$(open_below "$old")" ] && { echo "REFUSED    $l: a process has a file open below $old"; open_below "$old"; bad=1; continue; }
          fi
          mkdir -p "$(dirname "$l")"; ln -sfn "$t" "$l"; echo "linked     $l -> $t (was $s)"
        else echo "$s    $l -> $t"; bad=1; fi ;;
      *) echo "$s    $l -> $t"; bad=1 ;;
    esac
  done <<< "$table"
  return $bad
}
if [ "${1:-}" = "--selftest" ]; then
  tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
  mkdir -p "$tmp/repo/proj/data/cache" "$tmp/store/C/a" "$tmp/store/K"
  export LS_REPO="$tmp/repo" LS_STORE="$tmp/store" LS_TABLE="proj/data/cache/feats|C/a
proj/data/ckpts|K
proj/data/other|Z/nothing"
  me="$here/link_storage.sh"
  out="$(bash "$me" --check 2>&1)" && { echo "selftest FAILED: check passed on missing links"; exit 1; }
  echo "$out" | grep -q "^missing    .*feats" && echo "$out" | grep -q "^no target    .*other" || { echo "selftest FAILED: missing / no target not reported: $out"; exit 1; }
  bash "$me" --apply >/dev/null 2>&1 && { echo "selftest FAILED: apply must report the link without a target"; exit 1; }
  [ -L "$tmp/repo/proj/data/cache/feats" ] && [ -L "$tmp/repo/proj/data/ckpts" ] || { echo "selftest FAILED: apply did not link"; exit 1; }
  [ "$(readlink "$tmp/repo/proj/data/ckpts")" = "$tmp/store/K" ] || { echo "selftest FAILED: wrong link target"; exit 1; }
  export LS_TABLE="proj/data/cache/feats|C/a
proj/data/ckpts|K"
  bash "$me" --check >/dev/null || { echo "selftest FAILED: check must be clean after apply"; exit 1; }
  mv "$tmp/store/K" "$tmp/store/K2"
  out="$(bash "$me" --check 2>&1)" && { echo "selftest FAILED: a dangling link passed"; exit 1; }
  echo "$out" | grep -q "^dangling    .*ckpts" || { echo "selftest FAILED: dangling not reported: $out"; exit 1; }
  export LS_TABLE="proj/data/cache/feats|C/a
proj/data/ckpts|K2"
  out="$(bash "$me" --check 2>&1)" && { echo "selftest FAILED: a wrong target passed"; exit 1; }
  echo "$out" | grep -q "^wrong    .*ckpts" || { echo "selftest FAILED: wrong target not reported: $out"; exit 1; }
  bash "$me" --apply >/dev/null && bash "$me" --check >/dev/null || { echo "selftest FAILED: apply did not re-point"; exit 1; }
  mkdir -p "$tmp/repo/proj/data/real"; echo x > "$tmp/repo/proj/data/real/f"
  export LS_TABLE="proj/data/real|K2"
  out="$(bash "$me" --apply 2>&1)" && { echo "selftest FAILED: a real folder passed"; exit 1; }
  echo "$out" | grep -q "^real    " && [ -f "$tmp/repo/proj/data/real/f" ] && [ ! -L "$tmp/repo/proj/data/real" ] || { echo "selftest FAILED: a real folder was touched: $out"; exit 1; }
  echo "selftest ok: check reports missing / no target / dangling / wrong, apply links and re-points, a real folder is never replaced"
  exit 0
fi
case "${1:---check}" in
  --check) run check ;;
  --apply) run apply ;;
  *) echo "usage: link_storage.sh [--check | --apply | --selftest]"; exit 2 ;;
esac
