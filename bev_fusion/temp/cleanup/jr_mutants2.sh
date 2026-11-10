#!/bin/bash
# plant bugs into jobrunner one at a time; the selftest must fail for each; the files are restored afterwards
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
try "holdout inside args not looked for" api.py 'if not R.is_holdout(split, holdout, aliases) and any(' 'if False and any('
try "ledger card ignored for a diagnostic" api.py 'if row and (row.get("prediction")' 'if False and row and (row.get("prediction")'
try "done is read as failed" api.py 'if st in ("done", "ok"):
                return "done", ""' 'if False:
                return "done", ""'
try "earlier result deleted" api.py 'os.replace(result, self._jd(i, f"result.{k}.json"))' 'os.remove(result)'
try "blocked is not restartable" api.py 'LIVE = ("queued", "running", "held")' 'LIVE = ("queued", "running", "held", "blocked")'
try "dead stalled counts as alive" api.py 'return bool(self.R.command_alive({**rec, "status": "running"}) or self.R.pid_alive(rec.get("launcher_pid")))' 'return True'
try "launch keys silently ignored" api.py 'if launch:
            problems += [f"{k} has no effect' 'if False:
            problems += [f"{k} has no effect'
try "run folder form not mapped" api.py 'same = sorted(n for n in names if fold(n) == fold(name))' 'same = []'
try "finished jobs conflict forever" api.py 'if row["id"] != exclude and (row["state"] in LIVE or row["state"] == "planned" or (row["state"] == "stalled" and self._stalled_alive(row["id"]))):' 'if row["id"] != exclude:'
try "foreign run adopted" api.py 'if os.path.exists(os.path.join(self._rdir(jid), "run.json")):' 'if False:'
try "lock of a live owner ignored" api.py 'if owner and owner != os.getpid() and _alive(owner):' 'if False:'
try "run() always ok" api.py 'st["ok"] = st["state"] == "done"' 'st["ok"] = True'
try "twin run folder ignored" api.py 'twin = R.twin_refusal(self.records, jid)' 'twin = ""'
try "tuples not normalised" api.py 'args = json.loads(json.dumps(args))                            # what the child will read: tuples are lists' 'pass'
try "needs as text allowed" catalogue.py 'elif isinstance(v, bool) or not isinstance(v, (int, float))' 'elif isinstance(v, bool)'
try "JOBS of listed suites ignored" catalogue.py ' | set((cfg.get("jobrunner") or {}).get("suites") or [])' ''
try "TEST gate of the log skipped" api.py 'if R.is_holdout(split, holdout, aliases):
            log_path' 'if False:
            log_path'
restore
cmp jobrunner/api.py /tmp/jr_api_backup.py && cmp jobrunner/catalogue.py /tmp/jr_cat_backup.py && echo "restored identical"
