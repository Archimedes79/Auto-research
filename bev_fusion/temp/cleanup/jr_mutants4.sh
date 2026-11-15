#!/bin/bash
# plant bugs into jobrunner one at a time (the third round: results as arguments, deep check, new attempts); the selftest must fail for each; the files are restored afterwards
cd /mnt/c/Auto-research/autoresearch || exit 1
export TMPDIR=/tmp/jr_mine_mut; rm -rf "$TMPDIR"; mkdir -p "$TMPDIR"
for f in api catalogue call; do cp jobrunner/$f.py /tmp/jr_mine_mut_$f.bak; done
restore() { for f in api catalogue call; do cp /tmp/jr_mine_mut_$f.bak jobrunner/$f.py; done; }
trap restore EXIT
try() {  # try <name> <file> <old> <new>
  restore
  python3 - "jobrunner/$2" "$3" "$4" <<'EOF'
import sys
f, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(f).read()
if old not in s:
    sys.exit(3)
open(f, "w").write(s.replace(old, new, 1))
EOF
  if [ $? -ne 0 ]; then echo "$1: MUTANT DID NOT APPLY"; return; fi
  if python3 -m jobrunner selftest >$TMPDIR/mut.txt 2>&1; then echo "$1: SURVIVED"; else echo "$1: killed ($(grep -E 'AssertionError|Error:' $TMPDIR/mut.txt | tail -1 | cut -c1-100))"; fi
}
try "results of other jobs not filled in" call.py 'args = resolve_refs(args, a.jobs_dir)' 'args = args'
try "result argument gives no dependency" api.py 'after += [x for x in ref_after if x not in after and x != jid]' 'pass'
try "deep check ignored" api.py 'if deep and p["entry"] and not p["problems"]:' 'if False:'
try "dry run gets all arguments" call.py 'if a.only_known:' 'if False:'
try "no new attempt with changed arguments" api.py 'if st in ("failed", "blocked", "stopped", "planned", "unknown") or' 'if False or'
try "earlier attempt deleted instead of kept" api.py 'os.replace(self._jd(s["id"], f"{name}.json"), self._jd(s["id"], f"{name}.{k}.json"))' 'os.remove(self._jd(s["id"], f"{name}.json"))'
try "hypothesis of a diagnostic carries the arguments" api.py "f\"--hyp={spec['note'] or spec['call']}\", f\"--basis={summary}\"" "f\"--hyp={spec['note'] or summary}\""
try "queued job does not say what it waits for" api.py 'return "queued", beat[1] if len(beat) > 1 else ""' 'return "queued", ""'
try "dry run without a time limit" api.py 'timeout=DRY_TIMEOUT_S, stdin' 'timeout=None, stdin'
try "result arguments type-checked" api.py '{k for k, v in args.items() if C.refs_in(v)})' 'set())'
try "result of a job that does not exist accepted" api.py 'problems.append(f"$result {r}: no such job' 'pass  # problems.append(f"$result {r}: no such job'
restore
for f in api catalogue call; do cmp jobrunner/$f.py /tmp/jr_mine_mut_$f.bak || echo "NOT RESTORED $f"; done; echo "restored identical"
