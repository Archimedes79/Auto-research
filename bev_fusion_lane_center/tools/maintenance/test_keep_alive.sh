#!/bin/bash
# Test of keep_alive.sh: stub jobs in a temporary folder (no GPU, no run.py), sleeps shortened; textual mutants of the file must fail it.
# usage: bash tools/maintenance/test_keep_alive.sh
HERE="$(cd "$(dirname "$0")" && pwd)"
LIB="$HERE/keep_alive.sh"
fail() { echo "FAIL: $*"; exit 1; }

check() {  # LIBFILE -> 0 when every case holds
  local lib=$1 d; d=$(mktemp -d)
  (
    cd "$d" || exit 2
    export KA_RUNS=runs KA_POLL=0.1 KA_SLEEP=0.1
    source "$lib"
    ka_init "$d/log/chain.log" tst "ka_test_$$"
    mkdir -p runs/job
    # a job that ends 'done' on its 3rd launch
    cat > stub.sh <<'EOS'
#!/bin/bash
n=$(( $(cat launches 2>/dev/null || echo 0) + 1 )); echo $n > launches
[ "$n" -ge "${DONE_AT:-3}" ] && echo "10:00:00 done job" >> runs/job/log.txt
exit 1
EOS
    ka_keep RUN.1 job 6 bash stub.sh || { echo "case 1: keep returned 1"; exit 3; }
    [ "$(cat launches)" = 3 ] || { echo "case 1: launches $(cat launches), want 3"; exit 3; }
    grep -q "RUN.1 launch 3" log/chain.log && grep -q "RUN.1 done" log/chain.log || { echo "case 1: log lines"; exit 3; }
    # already done: nothing is launched
    ka_keep RUN.1 job 6 bash stub.sh; [ "$(cat launches)" = 3 ] || { echo "case 2: a finished job was launched again"; exit 4; }
    # gives up after MAX launches and says so
    rm -f launches runs/job/log.txt
    DONE_AT=99 ka_keep RUN.2 job 2 bash stub.sh && { echo "case 3: no give-up"; exit 5; }
    [ "$(cat launches)" = 2 ] && grep -q "RUN.2: 2 launches, giving up" log/chain.log || { echo "case 3: launches / log"; exit 5; }
    # a line of another run or a prefix of the name is not 'done'
    rm -f launches runs/job/log.txt; echo "10:00:00 done job2" > runs/job/log.txt; echo "10:00:00 xdone job" >> runs/job/log.txt
    DONE_AT=1 ka_keep RUN.3 job 3 bash stub.sh; [ "$(cat launches)" = 1 ] || { echo "case 4: launches $(cat launches), want 1 (only an exact 'done job' line counts)"; exit 6; }
    # a live `run.py --id RUN.4 ` of another supervisor: nothing is launched until its log says done
    rm -f launches runs/job/log.txt
    bash -c 'sleep 1; echo "10:00:00 done job" >> runs/job/log.txt' run.py --id RUN.4 x &
    DONE_AT=1 ka_keep RUN.4 job 3 bash stub.sh; wait
    [ ! -f launches ] || { echo "case 4b: launched while the other supervisor's job was alive"; exit 6; }
    # the window, over midnight and not
    KA_NOW=2300 ka_in_window 2200 0700 || { echo "case 5a"; exit 7; }
    KA_NOW=0300 ka_in_window 2200 0700 || { echo "case 5b"; exit 7; }
    KA_NOW=0700 ka_in_window 2200 0700 && { echo "case 5c: the end is outside"; exit 7; }
    KA_NOW=2159 ka_in_window 2200 0700 && { echo "case 5d"; exit 7; }
    KA_NOW=1200 ka_in_window 0830 1730 || { echo "case 5e"; exit 7; }
    KA_NOW=1730 ka_in_window 0830 1730 && { echo "case 5f"; exit 7; }
    KA_NOW=0805 ka_in_window 0800 0900 || { echo "case 5g: leading zeros (octal)"; exit 7; }
    KA_NOW=2300 ka_window 2200 0700
    grep -q "window open" log/chain.log || { echo "case 6: window"; exit 7; }
    # a second supervisor with the same lock exits at once (fd 8 of this shell holds it)
    bash -c 'cd "'"$d"'"; source "'"$lib"'"; ka_init "'"$d"'/log/chain.log" tst2 "ka_test_'"$$"'"; echo SECOND-RAN' | grep -q SECOND-RAN && { echo "case 7: the second supervisor ran"; exit 8; }
    grep -q "supervisor already runs" log/chain.log || { echo "case 7: log"; exit 8; }
    exit 0
  )
  local rc=$?
  rm -rf "$d"
  return $rc
}

check "$LIB" || fail "the original"
mutate() {  # NAME OLD NEW
  local f; f=$(mktemp); local src; src=$(cat "$LIB")
  [[ "$src" == *"$2"* ]] || fail "mutant '$1': text not found"
  printf '%s\n' "${src/"$2"/"$3"}" > "$f"
  if check "$f" > /dev/null 2>&1; then rm -f "$f"; fail "mutant '$1' survived"; fi
  rm -f "$f"
}
mutate "done not anchored" '^[0-9:]* done $name\$' '[0-9:]* done $name'
mutate "no give-up" '[ "$n" -gt "$max" ] && { say "$id: $max launches, giving up"; return 1; }' ':'
mutate "window end inclusive" '[ "$now" -lt "$e" ]; else' '[ "$now" -le "$e" ]; else'
mutate "octal clock" 'now=$((10#$now))' 'now=$((now))'
mutate "no lock" 'flock -n 8 ||' 'true ||'
mutate "launch while a job lives" 'if pgrep -f "run.py --id $id " > /dev/null; then sleep "$KA_POLL"; continue; fi' ':'
mutate "relaunch after done" 'while ! grep -q' 'while grep -q'
echo "keep_alive: all cases hold, 7 mutants killed"
