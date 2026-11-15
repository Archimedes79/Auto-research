#!/bin/bash
# plant bugs into jobrunner one at a time (the second review round); the selftest must fail for each; the files are restored afterwards
cd /mnt/c/Auto-research/autoresearch || exit 1
cp jobrunner/api.py /tmp/jr_api_backup.py
cp jobrunner/catalogue.py /tmp/jr_cat_backup.py
restore() { cp /tmp/jr_api_backup.py jobrunner/api.py; cp /tmp/jr_cat_backup.py jobrunner/catalogue.py; }
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
  if python3 -m jobrunner selftest >/tmp/jr_mut.txt 2>&1; then echo "$1: SURVIVED"; else echo "$1: killed ($(grep -E 'AssertionError|Error:' /tmp/jr_mut.txt | tail -1 | cut -c1-100))"; fi
}
try "free text scanned for the holdout" api.py 'scanned = [s for k, v in args.items() if k in names for s in _strings(v)]' 'scanned = [s for s in _strings(args)]'
try "ledger card rule blocks launch jobs" api.py 'card is None and not launch:' 'card is None:'
try "no second look under the lock" api.py '            if cur and skip():
                return skip()' '            pass'
try "launch method runs without a pid" api.py 'launcher_pid=os.getpid() if how == "launch" else None' 'launcher_pid=None'
try "no window for a restarted launch job" api.py 'if old or (job.get("how") == "launch" and rec.get("created", "") and rec["created"] < job.get("started_at", "") and age < 120):' 'if old:'
try "zombie launcher counts as alive" api.py 'or _alive(rec.get("launcher_pid")))      # _alive' 'or self.R.pid_alive(rec.get("launcher_pid")))      # _alive'
try "run.py options in run_args allowed" api.py 'if isinstance(t, str) and t.split("=", 1)[0] in RESERVED_RUN_ARGS]' 'if False]'
try "after is not checked" api.py 'elif self._find_run(x) is None:' 'elif False:'
try "signature asked every time" api.py 'if key not in self._sigs:' 'if True:'
try "a changed run.py is not noticed" api.py '            if missing:
                raise JobError("run.py has changed' '            if False:
                raise JobError("run.py has changed'
try "problems around a done job refuse it" api.py 'if p["existing"] and skip():' 'if False:'
try "archive overwritten on the second restart" api.py 'while os.path.exists(self._jd(i, f"result.{k}.json")):' 'while False:'
try "gpu false with a memory gate allowed" api.py 'if needs.get("gpu") is False and needs.get("gpu_max_used_mb"):' 'if False:'
try "own name as a dependency" api.py 'if rid == jid or (jid and self.R.run_folder(name.strip().rstrip("/")).lower() == self.R.run_folder(jid).lower()):' 'if False:'
try "lock owner may be a reused pid" api.py '_alive(owner, start or None)' '_alive(owner)'
try "splits key unknown" catalogue.py 'ENTRY_KEYS = ("suite", "cls", "method", "how", "may_feed_verdict", "needs", "reads", "writes", "python", "id_from", "splits")' 'ENTRY_KEYS = ("suite", "cls", "method", "how", "may_feed_verdict", "needs", "reads", "writes", "python", "id_from")'
restore
cmp jobrunner/api.py /tmp/jr_api_backup.py && cmp jobrunner/catalogue.py /tmp/jr_cat_backup.py && echo "restored identical"
