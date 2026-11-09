#!/bin/bash
# D45 (Mirko 2026-11-02): delete exactly: the 70-step checkpoints of speed_a / a3 / b / c2, the step-0 checkpoint of the held profile.t10, its code/ and bundle/ copies
# (only if equal to the tag bf/profile.t10), and the scratch config temp/compose/profile_t10.json. Text records stay. Prints what went.
cd /mnt/c/Auto-research/bev_fusion || exit 1
if pgrep -f "speed_[abc]|profile_t10|profile.t10" > /dev/null; then echo "a process names one of my targets: stop here"; pgrep -af "speed_[abc]|profile_t10|profile.t10" | cut -c1-140; exit 1; fi
total=0
del() { [ -f "$1" ] || { echo "not there: $1"; return; }; b=$(stat -c %s "$1"); rm -- "$1" && total=$((total+b)) && echo "deleted $1 ($((b/1048576)) MB)"; }
for r in speed_a speed_a3 speed_b speed_c2; do del runs/$r/ckpt_00000070.pt; done
del runs/profile_t10/ckpt_00000000.pt
# code/ of profile_t10: equal to the tag?
tag=bf/profile.t10
same=1
while read -r f; do
  rel=${f#./}
  case "$rel" in __pycache__/*|*/__pycache__/*|SNAPSHOT.txt) continue;; esac
  git -C .. show "$tag:bev_fusion/training/$rel" 2>/dev/null | cmp -s - "runs/profile_t10/code/$rel" || { echo "  code/$rel differs from $tag"; same=0; }
done < <(cd runs/profile_t10/code && find . -type f)
if [ "$same" = 1 ]; then n=$(du -sb runs/profile_t10/code | cut -f1); rm -rf runs/profile_t10/code && total=$((total+n)) && echo "deleted runs/profile_t10/code ($((n/1024)) KB, equal to the tag)"; else echo "KEPT runs/profile_t10/code"; fi
# bundle/: the network and loss files of the tag?
same=1
for f in results/profile.t10/bundle/*; do
  [ -f "$f" ] || continue
  b=$(basename "$f")
  git -C .. show "$tag:bev_fusion/training/$b" 2>/dev/null | cmp -s - "$f" || { echo "  bundle/$b differs from $tag"; same=0; }
done
if [ "$same" = 1 ]; then n=$(du -sb results/profile.t10/bundle | cut -f1); rm -rf results/profile.t10/bundle && total=$((total+n)) && echo "deleted results/profile.t10/bundle ($((n/1024)) KB, equal to the tag)"; else echo "KEPT results/profile.t10/bundle"; fi
del temp/compose/profile_t10.json
echo "freed $((total/1048576)) MB"
echo "--- what stays:"; ls runs/speed_a runs/speed_b runs/profile_t10 results/profile.t10 | tr '\n' ' '; echo
