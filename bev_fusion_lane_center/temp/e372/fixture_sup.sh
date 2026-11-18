#!/bin/bash
# E372 S4 data path: the GPU parity fixture of `lane_live_tok.py` (DIAGNOSTIC, run.py --no-card, feeds no verdict; the acceptance rule is the block 'GPU FIXTURE ACCEPTANCE' of that module's docstring, written before this run).
# Waits for the night GPU window (22:00 - 07:00; the evening is not for the GPU), then runs the fixture on the GPU with the B' of both seeds on the first 200 VAL frames: first bs 4 = the cache build's (the parity: exit 1 unless exact),
# then bs 8 = the joint trainer's batch size (information: the batch-size dependence of the GPU; a difference there is not a bug). Short jobs (a few minutes): they share a GPU slot with whatever the night supervisors run (run.py's FIFO slots).
# usage: bash temp/e372/fixture_sup.sh      log: temp/e372/chain.log, output: temp/e372/fixture_bs{4,8}.out, json: runs/e372_fixture/fixture_gpu_bs{4,8}.json
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
L=temp/e372/chain.log
say() { echo "$(date '+%m-%d %H:%M:%S') [e372 fixture] $*" >> $L; }
exec 8>/tmp/e372_fixture.lock
flock -n 8 || { say "supervisor already runs"; exit 0; }
mkdir -p runs/e372_fixture
say "waiting for the night window (22:00)"
until [ "$(date +%H%M)" -ge 2200 ] || [ "$(date +%H%M)" -lt 0700 ]; do sleep 30; done
say "night window open"
for BS in 4 8; do
  if grep -q "FIXTURE bs $BS " temp/e372/fixture_bs$BS.out 2>/dev/null; then say "bs $BS already done"; continue; fi
  say "bs $BS start"
  python3 ../autoresearch/tools/run.py --id E372.fixture_gpu_bs$BS --seed 0 --gpu --ram-mb 6000 --retries 0 --no-card --split val --timeout 40 -- bash tools/env/_pyr.sh lane_live_tok.py fixture --split val --n 200 --seeds 0 1 --gpu --bs $BS --out runs/e372_fixture/fixture_gpu_bs$BS.json >> temp/e372/fixture_bs$BS.out 2>&1
  say "bs $BS exit $?"
done
say "done"
