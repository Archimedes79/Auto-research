p = "temp/e291/run_staged_gpu.sh"
s = open(p, encoding="utf-8").read()
old = '''export BFLC_STAGE="${BFLC_STAGE:-$R/temp/_session_scratch_e291/stage}"
[ "$BFLC_STAGE" = "none" ] || [ -f "$BFLC_STAGE/training/grid/jends.py" ] || { echo "run_staged_gpu.sh: no staged jends.py in $BFLC_STAGE: this run would use the REAL files" >&2; exit 2; }
'''
new = '''STAGE_OK="$R/temp/_session_scratch_e291/stage"
if [ "$E291_TEST" = "1" ]; then                       # run_staged.sh (the CPU tests, the mutants): any stage, `none` = the real files
  export BFLC_STAGE="${BFLC_STAGE:-$STAGE_OK}"
  [ "$BFLC_STAGE" = "none" ] || [ -f "$BFLC_STAGE/training/grid/jends.py" ] || { echo "run_staged_gpu.sh: no staged jends.py in $BFLC_STAGE: this run would use the REAL files" >&2; exit 2; }
else                                                  # a training / dump: ONLY this stage (E347's wrappers use the same variable name: a shell that exports E347's stage must not start an E291 arm)
  [ -z "$BFLC_STAGE" ] || [ "$BFLC_STAGE" = "$STAGE_OK" ] || { echo "run_staged_gpu.sh: BFLC_STAGE=$BFLC_STAGE is not $STAGE_OK (another chat's stage?): REFUSED" >&2; exit 2; }
  export BFLC_STAGE="$STAGE_OK"
  [ -f "$BFLC_STAGE/training/grid/jends.py" ] || { echo "run_staged_gpu.sh: no staged jends.py in $BFLC_STAGE: this run would use the REAL files" >&2; exit 2; }
fi
'''
assert old in s
s = s.replace(old, new)
open(p, "w", encoding="utf-8", newline="\n").write(s)
p = "temp/e291/run_staged.sh"
s = open(p, encoding="utf-8").read()
s = s.replace('export CUDA_VISIBLE_DEVICES=""', 'export E291_TEST=1 CUDA_VISIBLE_DEVICES=""')
open(p, "w", encoding="utf-8", newline="\n").write(s)
print(s)
