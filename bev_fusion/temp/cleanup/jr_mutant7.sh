#!/bin/bash
cd /mnt/c/Auto-research/autoresearch || exit 1
F=jobrunner/api.py
cp $F /tmp/jr_api_backup7.py
trap 'cp /tmp/jr_api_backup7.py $F' EXIT
echo "--- selftest on the code as it is"
python3 -m jobrunner selftest > /tmp/jr_plain7.txt 2>&1; echo "rc=$?"; tail -n 12 /tmp/jr_plain7.txt | cut -c1-200
echo "--- mutant: lock owner may be a reused pid"
sed -i 's/_alive(owner, start or None)/_alive(owner)/' $F
grep -c "_alive(owner)" $F
python3 -m jobrunner selftest > /tmp/jr_mut7.txt 2>&1; echo "rc=$?"; tail -n 8 /tmp/jr_mut7.txt | cut -c1-200
cp /tmp/jr_api_backup7.py $F
cmp $F /tmp/jr_api_backup7.py && echo "restored identical"
