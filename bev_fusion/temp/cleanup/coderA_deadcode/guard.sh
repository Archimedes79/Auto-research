#!/bin/bash
# scratch: kill MY safety.py jobs if free RAM drops under 2 GB (other chats' jobs are never touched); exits when no safety.py is left
sleep 20
while pgrep -f "[d]eadcode/safety.py" > /dev/null; do
  avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  if [ "$avail" -lt 2000 ]; then
    echo "GUARD: MemAvailable ${avail} MB < 2000, killing safety.py"
    pkill -f "[d]eadcode/safety.py"
    exit 2
  fi
  sleep 3
done
echo "GUARD: safety.py jobs finished"
