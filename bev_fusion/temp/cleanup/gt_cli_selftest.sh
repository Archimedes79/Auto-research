#!/bin/bash
# run the GtGenerator command-line selftest from an empty working directory and list what it leaves there
W=/tmp/gtcli_cwd_mine; rm -rf "$W"; mkdir -p "$W"; cd "$W" || exit 1
bash /mnt/c/Auto-research/GtGenerator/run_nice.sh -m GtGenerator.cli_selftest > /tmp/gtcli_mine.txt 2>&1; rc=$?
echo "rc=$rc"; tail -n 3 /tmp/gtcli_mine.txt | cut -c1-200
echo "left in the working directory:"; ls -A "$W" | cat; echo "(end of list)"
