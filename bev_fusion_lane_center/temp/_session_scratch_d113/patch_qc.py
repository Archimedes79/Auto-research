import os
p = os.path.join(os.environ["BFLC_ROOT"], "temp", "e379", "quiet_chain.sh")
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
rep('''  $RUN --id "$id" --seed 0 --gpu --ram-mb 6000 --retries 0 $card --split val --timeout "$to" -- "$@" >> temp/e379/quiet_chain.out 2>&1
  say "$id exit $?"''', '''  if [ "$card" = "--card-e2" ]; then
    $RUN --id "$id" --seed 0 --gpu --ram-mb 6000 --retries 0 --require-card --hyp "$HYP" --pred "$PRED" --rule "$RULE" --split val --timeout "$to" -- "$@" >> temp/e379/quiet_chain.out 2>&1
  else
    $RUN --id "$id" --seed 0 --gpu --ram-mb 6000 --retries 0 $card --split val --timeout "$to" -- "$@" >> temp/e379/quiet_chain.out 2>&1
  fi
  say "$id exit $?"''')
rep('job E379.time_e2 20 "--require-card --hyp \\"$HYP\\" --pred \\"$PRED\\" --rule \\"$RULE\\"" bash', 'job E379.time_e2 20 --card-e2 bash')
rep('''  while ! gap; do sleep 30; done                                   # each job alone''', '''  while [ -z "$FORCE" ] && ! gap; do sleep 30; done                # each job alone''')
rep('''  if in_window && gap; then n=$((n + 1)); else n=0; fi''', '''  if [ -n "$FORCE" ] || { in_window && gap; }; then n=$((n + 1)); else n=0; fi''')
rep("(DRY=1: only say what it would run)", "(DRY=1: only say what it would run; FORCE=1: skip the window and gap checks, for the dry run)")
open(p, "w", encoding="utf-8").write(s)
print("ok")
