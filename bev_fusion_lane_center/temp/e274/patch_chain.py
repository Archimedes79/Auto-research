p = "temp/e274/chain_after_p.sh"
s = open(p).read()
old = s[s.index('echo "== sweeps'):]
new = '''echo "== sweeps (hard stop 15:20 for the BEV chat) $(date)" >> $L
left() { echo $(( $(date -d 15:20 +%s) - $(date +%s) )); }
run_sweep() { t=$(left); if [ "$t" -gt 120 ]; then timeout ${t}s bash tools/env/_py14.sh sweep_gpu.py "$@" >> $L 2>&1; else echo "skipped (past 15:20): $*" >> $L; fi; }
run_sweep e273_p_s1
run_sweep e273_p_s0 --skip-plain
run_sweep e274_r_s0 --skip-plain
run_sweep e274_r_s1 --skip-plain
echo "chain done $(date)" >> $L
touch temp/e274/CHAIN_DONE
'''
open(p, "w").write(s.replace(old, new))
