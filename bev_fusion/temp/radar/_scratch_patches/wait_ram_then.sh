#!/usr/bin/env bash
# waits until >= 9000 MB of RAM are available on three checks 20 s apart (the shared box swapped out at 22.5 of 24 GB), then runs the command
ok=0
for i in $(seq 1 720); do
  avail=$(free -m | awk '/^Mem:/ {print $7}')
  if [ "${avail:-0}" -ge 9000 ]; then ok=$((ok + 1)); else ok=0; fi
  [ "$ok" -ge 3 ] && break
  sleep 20
done
echo "$(date +%T) RAM available ${avail} MB (checks $ok): starting $*"
"$@"
