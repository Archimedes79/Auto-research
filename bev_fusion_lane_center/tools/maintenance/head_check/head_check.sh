#!/bin/bash
# The clean-copy check of a commit before a push (used before the pushes of 2026-11-18, 11-22 and 11-23; promoted from scratch on 2026-11-23, D140):
# `git archive <ref>` of the whole repository is unpacked OUTSIDE the repository (~/head_check/<sha>: no data, no runs/, no untracked file), and there, CPU only
# (CUDA hidden, 2 threads, nice 10), one PASS / FAIL line each:
#   the token tests, the live / cache tests, jarm, pmap, the check_git / commit_paths selftests          (the champion's code and the git tools)
#   archdraw check of the lane drawing, check_structure, the tests of the map line / route head / E393     (the rows the lane chat added on 2026-11-23)
#   the import audit: every import of every committed .py resolved in the copy (head_imports.py; a name found only in the LIVE tree = a file the commit lacks),
#   then each README-named module imported for real, one process each, 60 s each (head_import_one.sh + head_import1.py)
# Known FAILs that are data, not code (a clean copy has no runs/ and no caches): test_jlive_train, test_lane_live, test_lane_route; check_structure lists the dead
# paths of data and scratch files that a checkout does not carry. Read the NEW rows, not the count.
# usage (WSL, from anywhere):   bash tools/maintenance/head_check/head_check.sh [<ref>, default HEAD]      ONLY_IMPORTS=1: the import audit alone
#                               bash tools/maintenance/head_check/head_check.sh --selftest                syntax of the four files + one real import, no copy
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"                     # the workspace root (the git repository); the script finds it from its own place
if [ "${1:-}" = "--selftest" ]; then
  bash -n "$HERE/head_check.sh" && bash -n "$HERE/head_import_one.sh" || { echo "head_check selftest FAIL: shell syntax"; exit 1; }
  python3 -m py_compile "$HERE/head_imports.py" "$HERE/head_import1.py" || { echo "head_check selftest FAIL: python syntax"; exit 1; }
  P="$REPO/bev_fusion_lane_center"; T=$(mktemp -d); mkdir -p "$T/temp"; cp "$HERE/head_import1.py" "$T/temp/"
  out=$(BFLC_ROOT="$P" timeout 120 bash "$P/tools/env/_py14.sh" "$T/temp/head_import1.py" "$P/evaluation/pmap.py" 2>&1 | grep -E "^IMPORT")
  rm -r -- "$T"
  case "$out" in "IMPORT OK"*) echo "head_check selftest ok: the four files parse, one real import ($out)"; exit 0;; *) echo "head_check selftest FAIL: $out"; exit 1;; esac
fi
REF=${1:-HEAD}
SHA=$(git -C "$REPO" rev-parse --short=10 "$REF") || { echo "unknown ref $REF"; exit 2; }
C=$HOME/head_check/$SHA
rm -rf -- "$C"; mkdir -p "$C"
git -C "$REPO" archive "$SHA" | tar -x -C "$C" || { echo "git archive / tar failed"; exit 2; }
P=$C/bev_fusion_lane_center
mkdir -p "$P/temp" "$C/head_check_logs"
cp "$HERE/head_imports.py" "$HERE/head_import1.py" "$P/temp/"            # the audit runs from the copy (also for a ref older than this tool)
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY="nice -n 10 bash $P/tools/env/_py14.sh"
echo "clean copy of $REF = $SHA at $C ($(date '+%H:%M'))"
run() {                                                    # run <name> <command ...>: one PASS / FAIL line, the output in head_check_logs/<name>.log
  name="$1"; shift
  log="$C/head_check_logs/$(echo "$name" | tr ' /()' '____').log"
  "$@" > "$log" 2>&1; rc=$?
  last=$(grep -vE "^\s*$" "$log" | tail -1 | cut -c1-140)
  if [ $rc -eq 0 ]; then echo "PASS  $name  | $last"; else echo "FAIL  $name (rc $rc) | $last"; fi
}
if [ -z "${ONLY_IMPORTS:-}" ]; then
  cd "$P" || exit 2
  for t in grid/test_jtoken.py test_jtoken_eval.py grid/test_lane_tokcache.py grid/test_jlive_train.py grid/test_lane_live.py \
           grid/test_sdgraph.py grid/test_mapassoc.py grid/test_maperror.py grid/test_jroute.py grid/test_lane_route.py; do
    run "$t" $PY "$t"
  done
  run "jarm selftest" $PY jarm.py selftest
  run "pmap" $PY evaluation/pmap.py
  run "check_git selftest" $PY "$C/autoresearch/tools/check_git.py" --selftest
  run "commit_paths selftest" $PY "$C/autoresearch/tools/commit_paths.py" --selftest
  run "archdraw check (lane drawing)" python3 "$C/autoresearch/tools/archdraw.py" check "$P/training/architecture.json"
  run "check_structure (lane)" python3 "$C/autoresearch/tools/check_structure.py"
fi
run "head_imports_static" $PY "$P/temp/head_imports.py" "$REPO"
echo "head_imports summary:"; grep -E "^head_imports|NOT IN HEAD|README NAMES|unresolved anywhere" "$C/head_check_logs/head_imports_static.log" | cut -c1-260
xargs -a "$P/temp/head_named.txt" -P 3 -I{} bash "$HERE/head_import_one.sh" {} "$P" > "$C/head_check_logs/imports_real.log" 2>&1
echo "real imports: $(grep -c '^IMPORT OK' "$C/head_check_logs/imports_real.log") ok, $(grep -c '^IMPORT FAIL' "$C/head_check_logs/imports_real.log") failed, $(grep -c '^IMPORT TIMEOUT' "$C/head_check_logs/imports_real.log") timeout, $(grep -c '^IMPORT CRASH' "$C/head_check_logs/imports_real.log") crash (of $(wc -l < "$P/temp/head_named.txt") README-named)"
grep -E "^IMPORT (FAIL|TIMEOUT|CRASH)" "$C/head_check_logs/imports_real.log" | sort | cut -c1-260
echo "done $(date '+%H:%M'); logs in $C/head_check_logs/ (remove the copy with: rm -r ~/head_check/$SHA)"
