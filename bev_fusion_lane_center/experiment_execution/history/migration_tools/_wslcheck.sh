#!/bin/bash
echo "== symlinks into the project"
find ~ -maxdepth 4 -type l 2>/dev/null | while read l; do t=$(readlink "$l"); case "$t" in *ML-Experiment*) echo "$l -> $t";; esac; done | head -20
echo "== wsl-home scripts mentioning the project"
grep -l "ML-Experiment" ~/*.sh ~/.bashrc ~/.profile 2>/dev/null
echo "== sizes of clutter in wsl home"
du -sh ~/_review_v6_b ~/_review_v6_c ~/_verify_v4 ~/_verify_v5 ~/_verify_v6 ~/_verify_v7 2>/dev/null
ls -d ~/_* 2>/dev/null | tr '\n' ' '
echo
echo "== exp dirs"
ls ~/exp_gcar_b123 | head -30 | tr '\n' ' '
echo
ls ~/exp_gcar_b123/p17 2>/dev/null | head -5 | tr '\n' ' '
du -sh ~/exp_gcar_b123/* 2>/dev/null | sort -rh | head -12
echo "== data"
du -sh ~/data/* 2>/dev/null
echo "== env var / pycache"
find /mnt/c/ML-Experiment/scripts -name "__pycache__" -maxdepth 1 | head -2
