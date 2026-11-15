#!/bin/bash
# selftest on the code as it is, then two planted bugs for the after_from change; the files are restored afterwards
cd /mnt/c/Auto-research/autoresearch || exit 1
cp jobrunner/api.py /tmp/jr_api_b8.py; cp jobrunner/catalogue.py /tmp/jr_cat_b8.py
trap 'cp /tmp/jr_api_b8.py jobrunner/api.py; cp /tmp/jr_cat_b8.py jobrunner/catalogue.py' EXIT
python3 -m jobrunner selftest > /tmp/jr_b8_plain.txt 2>&1; echo "plain: rc=$? $(tail -n 1 /tmp/jr_b8_plain.txt | cut -c1-100)"; [ $? -eq 0 ] || tail -n 12 /tmp/jr_b8_plain.txt | cut -c1-200
m() {
  cp /tmp/jr_api_b8.py jobrunner/api.py; cp /tmp/jr_cat_b8.py jobrunner/catalogue.py
  python3 - "jobrunner/$2" "$3" "$4" <<'EOF'
import sys
f, old, new = sys.argv[1:4]
s = open(f).read()
if old not in s:
    sys.exit(3)
open(f, "w").write(s.replace(old, new, 1))
EOF
  [ $? -eq 0 ] || { echo "$1: DID NOT APPLY"; return; }
  python3 -m jobrunner selftest > /tmp/jr_b8.txt 2>&1 && echo "$1: SURVIVED" || echo "$1: killed ($(grep -E 'AssertionError' /tmp/jr_b8.txt | tail -1 | cut -c1-90))"
}
m "after_from not read" api.py 'for x in (C.ids_from(entry["after_from"], args) if entry and entry.get("after_from") else []):' 'for x in []:'
m "relative path message without the folder" api.py '(a relative path is read from the current folder, {os.getcwd()})' ''
m "launch call shows no in_process" api.py 'in_process=f"{p['"'"'entry'"'"']['"'"'cls'"'"']}(project)' 'in_process=None or f"{p['"'"'entry'"'"']['"'"'cls'"'"']}(project)'
cp /tmp/jr_api_b8.py jobrunner/api.py; cp /tmp/jr_cat_b8.py jobrunner/catalogue.py
cmp jobrunner/api.py /tmp/jr_api_b8.py && cmp jobrunner/catalogue.py /tmp/jr_cat_b8.py && echo "restored identical"
