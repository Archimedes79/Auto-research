p = "temp/day30/day30.sh"
s = open(p, encoding="utf-8").read()
old_job = s[s.index("# job <run id>"):s.index("stage() {")]
new_job = '''# okrun <run id>: the run record says the job finished with exit 0 (a chain killed between the end of a job and its .done marker must not start the id again: run.py refuses a finished id)
okrun() { python3 -c "
import json,sys
try: st=json.load(open('experiment_execution/runs/$1/run.json')).get('status')
except Exception: st=None
sys.exit(0 if st=='ok' else 1)"; }
# job <run id> <seed> <after ids or -> <command ...>: one run.py job (blocks until it ends); a job with a .done marker (or an 'ok' run record) is skipped. --timeout counts the wall time INCLUDING a pause: a SIGSTOP of the dump of up to about 100 minutes is harmless, a longer one times
# the job out (exit 124, the stage fails, the chain stops); starting day30.sh again resumes at that job (run.py resumes a failed id, the dump begins again: it writes only at its end)
job() {
  local id=$1 seed=$2 after=$3; shift 3
  [ -f $D/$id.done ] && return 0
  local af=""; [ "$after" != "-" ] && af="--after $after"
  if [ -n "$DAY30_DRY" ]; then say "DRY run.py --id $id --seed $seed $af --timeout $TIMEOUT -- nice -n 19 $*"; return 0; fi
  okrun $id && { touch $D/$id.done; return 0; }
  python3 ../autoresearch/tools/run.py --id $id --seed $seed --retries 0 --require-card --split val --timeout $TIMEOUT $af --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- nice -n 19 "$@" && touch $D/$id.done
}
'''
s = s.replace(old_job, new_job)
s = s.replace('THREADS=${DAY30_THREADS:-1}\n', 'THREADS=${DAY30_THREADS:-1}\nTIMEOUT=${DAY30_TIMEOUT:-120}\n')
s = s.replace("# Threads of the dump: DAY30_THREADS", "# SIGSTOP / SIGCONT of the python processes (the E354 gate): harmless, nothing is polled or timed inside them but run.py's wall-clock --timeout (DAY30_TIMEOUT minutes, default 120 per job; a dump takes 12 - 25 minutes); the chain itself is bash and keeps waiting.\n# Threads of the dump: DAY30_THREADS")
open(p, "w", encoding="utf-8", newline="\n").write(s)
