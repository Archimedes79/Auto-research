#!/bin/bash
# the selftest with a private temp folder of its own (other chats run selftests too: nothing outside it is touched)
cd /mnt/c/Auto-research/autoresearch || exit 1
MY=/tmp/jr_mine_keep
rm -rf "$MY"
mkdir -p "$MY"
TMPDIR=$MY JOBRUNNER_KEEP=1 python3 -m jobrunner selftest > $MY/out.txt 2>&1; echo "rc=$?"
tail -n 12 $MY/out.txt | cut -c1-240
T=$(ls -d $MY/jobrunner_selftest_* | head -1)
echo "T=$T"
[ -f $T/runs/.jobs/e.pk/launch.log ] && { echo "--- launch.log of e.pk"; head -6 $T/runs/.jobs/e.pk/launch.log | cut -c1-260; }
