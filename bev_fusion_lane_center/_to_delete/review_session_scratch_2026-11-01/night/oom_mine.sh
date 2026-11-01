# make my lane CPU arms (run.py E26* and everything under them) the OOM killer's first choice; children forked later inherit it
for root in $(pgrep -f "run.py --id E26"); do
  for p in $root $(pgrep -P $root) $(for c in $(pgrep -P $root); do pgrep -P $c; done) $(for c in $(pgrep -P $root); do for g in $(pgrep -P $c); do pgrep -P $g; done; done); do
    echo 1000 > /proc/$p/oom_score_adj 2>/dev/null && echo "$p $(tr '\0' ' ' < /proc/$p/cmdline | cut -c1-80)"
  done
done
