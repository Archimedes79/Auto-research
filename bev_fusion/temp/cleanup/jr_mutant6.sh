#!/bin/bash
cd /mnt/c/Auto-research/autoresearch || exit 1
F=jobrunner/catalogue.py
cp $F /tmp/jr_cat_backup.py
trap 'cp /tmp/jr_cat_backup.py $F' EXIT
python3 - <<'PY'
f = "jobrunner/catalogue.py"
s = open(f).read()
old = 'if not any(p["var"] for p in params):'
assert old in s
open(f, "w").write(s.replace(old, "if False:", 1))
PY
grep -c "if False:" $F
python3 -m jobrunner selftest 2>&1 | tail -3 | cut -c1-220
cp /tmp/jr_cat_backup.py $F
cmp $F /tmp/jr_cat_backup.py && echo "restored identical"
