# scratch: wait until a scratch output file has the line $2, then show it
f=${1:-temp/e379/e379f_live_tests.out}
until grep -q "^${2:-ALLDONE}" "$f" 2>/dev/null; do sleep 5; done
cat "$f"
