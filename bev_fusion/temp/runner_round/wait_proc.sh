#!/bin/bash
# scratch: wait (up to $2 x 10 s) until no process matches pattern $1 (given as [r]un_base_real.sh), then show the tails of the given logs
pat="$1"; n="${2:-58}"; shift 2
for ((i = 0; i < n; i++)); do
  pgrep -f "$pat" > /dev/null || break
  sleep 10
done
pgrep -f "$pat" > /dev/null && echo "still running"
for f in "$@"; do echo "=== $f"; tail -8 "$f" | cut -c1-240; done
