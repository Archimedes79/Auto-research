#!/bin/bash
# plant six bugs into jobrunner/api.py one at a time; the selftest must fail for each; the file is restored afterwards
cd /mnt/c/Auto-research/autoresearch || exit 1
F=jobrunner/api.py
cp $F /tmp/jr_api_backup.py
restore() { cp /tmp/jr_api_backup.py $F; }
trap restore EXIT
try() {  # try <name> <python expression that edits the source text s>
  restore
  python3 - "$F" "$2" <<'EOF'
import sys
f, expr = sys.argv[1], sys.argv[2]
s = open(f).read()
old, new = eval(expr)
assert old in s, "mutant does not apply: " + old
open(f, "w").write(s.replace(old, new, 1))
EOF
  if [ $? -ne 0 ]; then echo "$1: MUTANT DID NOT APPLY"; return; fi
  if python3 -m jobrunner selftest >/tmp/jr_mut.txt 2>&1; then echo "$1: SURVIVED (selftest passed)"; else echo "$1: killed ($(grep -E 'AssertionError|Error' /tmp/jr_mut.txt | tail -1 | cut -c1-110))"; fi
}
try "no --no-git for diagnostics" '("\"--no-card\", \"--no-git\"", "\"--no-card\"")'
try "reads do not give --after" '("after += [x for x in inferred if x not in after]", "pass")'
try "gpu_busy counts as done" '("return \"failed\", \"no GPU slot became free in time\"", "return \"done\", \"\"")'
try "same job started twice" '("if st in LIVE or st == \"done\":", "if False:")'
try "writes conflicts ignored" '("        if not writes or not os.path.isdir(self.jobs):\n            return []", "        return []")'
try "unknown arguments allowed" '("if not any(p[\"var\"] for p in params):", "if False:")'
restore
echo "restored: $(cmp $F /tmp/jr_api_backup.py && echo identical)"
