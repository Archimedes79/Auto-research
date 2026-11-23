#!/bin/bash
# one real import of the clean-copy check (tools/maintenance/head_check/head_check.sh): <module file> <copy problem folder>; at most 60 s (a module that works at import is killed and reported as TIMEOUT)
f=$1; P=$2
raw=$(timeout -k 5 60 bash "$P/tools/env/_py14.sh" "$P/temp/head_import1.py" "$f" 2>&1); rc=$?
out=$(echo "$raw" | grep -E "^IMPORT")
if [ -n "$out" ]; then echo "$out"
elif [ $rc -eq 124 ] || [ $rc -eq 137 ]; then echo "IMPORT TIMEOUT ${f#$P/} (works at import for more than 60 s)"
else echo "IMPORT CRASH ${f#$P/} (rc $rc): $(echo "$raw" | grep -v '^_py14' | tail -1 | cut -c1-160)"; fi
