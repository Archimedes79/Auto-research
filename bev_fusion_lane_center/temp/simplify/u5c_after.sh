#!/bin/bash
# U5 scratch: start u5c.sh when u5b.sh has finished (one test job of mine at a time on the shared box; max 40 min wait)
cd "$(dirname "$0")"
end=$((SECONDS + 2400))
until grep -q "u5b done" u5b.out 2>/dev/null || [ $SECONDS -ge $end ]; do sleep 20; done
bash u5c.sh > u5c.out 2>&1
cat u5b.out
grep -v "^u2_run" u5b_u_jlive_check.out | tail -n 20 | cut -c1-200
echo "--- u5c"
cat u5c.out
