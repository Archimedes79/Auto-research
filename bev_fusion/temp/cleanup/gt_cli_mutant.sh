#!/bin/bash
# planted bug: the old view.py (output folder created before the pick is checked) in a COPY of GtGenerator; the new test must fail
M=/tmp/gtmut_mine; rm -rf "$M"; mkdir -p "$M/GtGenerator"
cp -r /mnt/c/Auto-research/GtGenerator/. "$M/GtGenerator/"
(cd /mnt/c/Auto-research && git show HEAD:GtGenerator/view.py) > "$M/GtGenerator/view.py"
W=/tmp/gtmut_cwd_mine; rm -rf "$W"; mkdir -p "$W"; cd "$W" || exit 1
PYTHONPATH="$M" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 15 ~/miniconda3/envs/gcar/bin/python -m GtGenerator.cli_selftest > /tmp/gtmut_mine.txt 2>&1; rc=$?
echo "mutant rc=$rc"; grep -E "AssertionError|Error" /tmp/gtmut_mine.txt | tail -n 2 | cut -c1-260; ls -A "$W" | cat
