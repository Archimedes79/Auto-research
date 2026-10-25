#!/bin/bash
# wait (max $1 minutes, default 8) until no test suite / launcher / python job of the project runs; print the tail of the suite log
for i in $(seq 1 $(( ${1:-8} * 12 ))); do
  ps -eo args | grep -E '[r]un_all_tests|[r]un\.py|[p]16_|[p]18_|evaluate\.py' | grep -qv grep || break
  sleep 5
done
tail -n ${2:-9} /tmp/rat_loop19.log
ps -eo etime,args | grep -E '[r]un_all_tests|[r]un\.py|[p]16_|[p]18_' | cut -c1-100
