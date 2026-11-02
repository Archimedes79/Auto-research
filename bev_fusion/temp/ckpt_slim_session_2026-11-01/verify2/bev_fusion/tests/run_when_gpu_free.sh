#!/usr/bin/env bash
# Wait until the GPU is free enough (memory used < 7000 MiB and utilisation < 50 % on 3 checks 30 s apart; at most 8 h), then run
# the given command. Never kills or touches another job. Usage: bash tests/run_when_gpu_free.sh <command ...>
ok=0
for i in $(seq 1 960); do
  read -r util mem < <(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ',' )
  if [ "${util:-100}" -lt 50 ] && [ "${mem:-99999}" -lt 7000 ]; then ok=$((ok + 1)); else ok=0; fi
  [ "$ok" -ge 3 ] && break
  sleep 30
done
[ "$ok" -ge 3 ] || { echo "$(date +%T) GPU stayed busy for 8 h, not started"; exit 1; }
echo "$(date +%T) GPU free (util $util %, $mem MiB): starting $*"
"$@"
