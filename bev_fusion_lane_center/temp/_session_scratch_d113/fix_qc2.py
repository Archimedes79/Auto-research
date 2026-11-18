import os
p = os.path.join(os.environ["BFLC_ROOT"], "temp", "e379", "quiet_chain.sh")
s = open(p, encoding="utf-8").read()
a = '''  while [ -z "$FORCE" ] && ! gap; do sleep 30; done                # each job alone: wait for the gap again (the previous job's wrapper may still be alive for a second)
'''
assert s.count(a) == 1
b = '''  while [ -z "$FORCE" ] && ! gap; do too_late "$id"; sleep 30; done  # each job alone: wait for the gap again (the previous job's wrapper may still be alive for a second)
  too_late "$id"
'''
s = s.replace(a, b)
c = '''RUN="python3 ../autoresearch/tools/run.py"
'''
assert s.count(c) == 1
d = '''RUN="python3 ../autoresearch/tools/run.py"
too_late() {  # the lead's rule (2026-11-18 02:50): NO new quiet-chain job after 06:45, the 07:00 landing has priority; what is left moves to the day window after the landing, on the landed tree
  local now=$((10#$(date +%H%M)))
  if [ -z "$FORCE" ] && [ $now -ge 645 ] && [ $now -lt 1200 ]; then say "$1: after 06:45: the chain stops here (the landing has priority); the rest runs in the day window after the landing"; exit 0; fi
}
'''
s = s.replace(c, d)
open(p, "w", encoding="utf-8").write(s)
print("ok")
