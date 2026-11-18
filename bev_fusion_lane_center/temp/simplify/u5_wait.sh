#!/bin/bash
# U5 scratch: wait (max 25 min) until u5b.sh has finished, then print its summary and the tail of the jlive check
cd "$(dirname "$0")"
end=$((SECONDS + 1500))
until grep -q "u5b done" u5b.out 2>/dev/null || [ $SECONDS -ge $end ]; do sleep 20; done
cat u5b.out
grep -v "^u2_run" u5b_test_jtoken.out | tail -n 3 | cut -c1-400
echo "--- jlive check"
grep -v "^u2_run" u5b_u_jlive_check.out | tail -n 60 | cut -c1-300
