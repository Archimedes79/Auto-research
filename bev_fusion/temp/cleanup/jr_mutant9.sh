#!/bin/bash
cd /mnt/c/Auto-research/autoresearch || exit 1
export TMPDIR=/tmp/jr_mine_mut9; rm -rf "$TMPDIR"; mkdir -p "$TMPDIR"
cp jobrunner/api.py /tmp/jr_mine_mut9_api.bak
trap 'cp /tmp/jr_mine_mut9_api.bak jobrunner/api.py' EXIT
python3 -m jobrunner selftest > $TMPDIR/plain.txt 2>&1; echo "plain: rc=$? $(tail -n 1 $TMPDIR/plain.txt | cut -c1-80)"
python3 - <<'EOF'
f = "jobrunner/api.py"
s = open(f).read()
old = "{k for k, v in args.items() if C.refs_in(v)})"
assert old in s
open(f, "w").write(s.replace(old, "set())", 1))
EOF
python3 -m jobrunner selftest > $TMPDIR/mut.txt 2>&1 && echo "result arguments type-checked: SURVIVED" || echo "result arguments type-checked: killed ($(grep -E 'AssertionError' $TMPDIR/mut.txt | tail -1 | cut -c1-110))"
cp /tmp/jr_mine_mut9_api.bak jobrunner/api.py; cmp jobrunner/api.py /tmp/jr_mine_mut9_api.bak && echo "restored identical"
