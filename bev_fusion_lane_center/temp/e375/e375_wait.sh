#!/bin/bash
# E375 scratch: wait until a log of this folder has its EXIT line (at most $2 x 10 s), then print its result lines
f="$1"; n="${2:-170}"
for i in $(seq 1 "$n"); do grep -q '^EXIT' "$f" && break; sleep 10; done
grep -E 'ok$|Error|EXIT|Traceback|BITFOR|files:|line [0-9]+' "$f" | cut -c1-900
