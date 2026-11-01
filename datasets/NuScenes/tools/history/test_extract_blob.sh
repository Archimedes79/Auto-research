#!/usr/bin/env bash
# Self-test of extract_blob.sh: (1) NUSC_WINTAR=1 + NUSC_MEMBERS=sweeps onto D: unpacks only sweeps/;
# (2) an archive with a '..' path is refused. Uses a scratch folder in the WSL home and on D:.
set -u
here="$(cd "$(dirname "$0")/.." && pwd)"
w="$HOME/tmp_extract_test"; d=/mnt/d/datasets/NuScenes/_extract_test
mkdir -p "$w/src/sweeps/CAM_TEST" "$w/src/samples/CAM_TEST" "$d"
echo a > "$w/src/sweeps/CAM_TEST/a.jpg"; echo b > "$w/src/samples/CAM_TEST/b.jpg"
tar -czf "$w/good.tgz" -C "$w/src" sweeps samples
NUSC_DEST="$d" NUSC_MEMBERS=sweeps NUSC_WINTAR=1 bash "$here/extract_blob.sh" "$w/good.tgz" > /dev/null
ok=1
[ -f "$d/sweeps/CAM_TEST/a.jpg" ] || { echo "FAIL: sweeps not unpacked"; ok=0; }
[ ! -e "$d/samples" ] || { echo "FAIL: samples unpacked although only sweeps was asked"; ok=0; }
mkdir -p "$w/evil/x"; echo e > "$w/evil/x/e.txt"
tar -czf "$w/bad.tgz" -C "$w/evil/x" --transform 's,^,../,' e.txt
if NUSC_DEST="$d" bash "$here/extract_blob.sh" "$w/bad.tgz" > /dev/null 2>&1; then echo "FAIL: unsafe archive accepted"; ok=0; fi
[ ! -e "$d/../e.txt" ] || { echo "FAIL: file escaped the target"; ok=0; }
mkdir -p /mnt/d/datasets/NuScenes/_to_delete "$HOME/_to_delete"
mv "$d" "/mnt/d/datasets/NuScenes/_to_delete/extract_test_$(date +%H%M%S)"; mv "$w" "$HOME/_to_delete/extract_test_$(date +%H%M%S)"
[ $ok = 1 ] && echo "extract_blob selftest: PASS" || { echo "extract_blob selftest: FAIL"; exit 1; }
