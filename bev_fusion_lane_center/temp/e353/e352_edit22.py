p = "temp/day30/day30.sh"
s = open(p, encoding="utf-8").read()
old = '''  for K in snap primary attach; do
    local line
    line=$(bash tools/env/_py14.sh jprune_b.py commands --seed $S --point $P $gf --readout $K 2>/dev/null | tail -1)
    [ -n "$line" ] || { say "no scoring line for $P $K"; return 1; }
    job ${id}_${K}_s$S $S ${id}_s$S $(eval echo "$line") || return 1
  done'''
new = '''  local lines line
  lines=$(bash tools/env/_py14.sh jprune_b.py commands --seed $S --point $P $gf 2>/dev/null)         # the scoring lines of the point (one python start)
  for K in snap primary attach; do
    line=$(echo "$lines" | grep -A1 "^# $P $K\$" | tail -1)
    case "$line" in "bash tools/env/_py14.sh jends_eval.py score "*) ;; *) say "no scoring line for $P $K"; return 1;; esac
    job ${id}_${K}_s$S $S ${id}_s$S $(eval echo "$line") || return 1
  done'''
assert old in s
open(p, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
