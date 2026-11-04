#!/usr/bin/env bash
# one-off: commit the vectorised radar reader through a temporary index: nuscenes_dataset.py whole (its diff is only this change, checked), training/README.md =
# HEAD + the dataset row only. Run from the repository root (Git Bash).
set -euo pipefail
T=bev_fusion/temp/arch_build
export GIT_INDEX_FILE=.git/index.radar
cp .git/index $GIT_INDEX_FILE
git read-tree HEAD
git add -- bev_fusion/training/nuscenes_dataset.py
r=bev_fusion/training/README.md
row=$(grep -F '| `nuscenes_dataset.py` |' $r)
git show HEAD:$r | awk -v row="$row" 'index($0, "| `nuscenes_dataset.py` |") == 1 { print row; next } { print }' > $T/readme_radar.md
[ "$(grep -c 'parse_radar_pcd' $T/readme_radar.md)" = 1 ]
git update-index --cacheinfo 100644,$(git hash-object -w $T/readme_radar.md),$r
git diff --cached --stat HEAD
msg='nuscenes_dataset: the radar sweeps are read together and parsed without the devkit loop (165 - 200 ms -> 28 ms per sample, identical output)

96 % of FrontRadarDataset._radar was RadarPointCloud.from_file: 32 files per sample opened one after the other (3 ms each on /mnt/c) and
every point and field unpacked in a Python loop. parse_radar_pcd reads the binary .pcd with one structured numpy view and applies the
devkit'"'"'s default filters (valid cluster, dynamic property 0 - 6, unambiguous Doppler; an empty cloud gives (18, 0)); read_files
reads the files of a sample with a small thread pool (per process, so a forked loader worker makes its own). The pose maths and the
output are untouched: the radar caches stay valid.
Checked: on 15 mini samples in two passes every one of the 8 output fields is torch.equal to the devkit-based function of HEAD (21 510
detections); the dataset self-test compares the parser with the devkit on 30 real files of all five radars and on an empty cloud;
test_bundle passes. Time on the mini split, CPU: old 165 - 201 ms, new 28 ms per sample. Matters for the radar-cache build, the
runner / precompute and evaluation without cache; a training with the cache already skipped this step.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>'
c=$(printf '%s\n' "$msg" | git commit-tree $(git write-tree) -p HEAD)
git update-ref refs/heads/main $c HEAD
rm -f $GIT_INDEX_FILE; unset GIT_INDEX_FILE
git reset -q -- bev_fusion/training/nuscenes_dataset.py $r
git log --oneline -1
