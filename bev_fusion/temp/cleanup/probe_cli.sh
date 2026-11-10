#!/bin/bash
# scratch: how the three suites behave on a cheap read-only call, a bad argument and --json (exit code, stdout head, stderr head)
cd /mnt/c/Auto-research/autoresearch || exit 1
export AR_PROJECT=/mnt/c/Auto-research/bev_fusion
P=~/miniconda3/envs/gcar/bin/python
t() {
  echo "--- $*"
  out=$("$@" 2>/tmp/probe_err); rc=$?
  echo "exit $rc | stdout: $(echo "$out" | head -c 160 | tr '\n' ' ')"
  echo "         stderr: $(head -c 200 /tmp/probe_err | tr '\n' ' ')"
}
t $P -m trainsuite Training.queue_list --json
t $P -m runsuite RunJobs.queue_list --json
t $P -m evalsuite Evaluation.sets --json
t $P -m evalsuite Runner.list runs/t20_base_s0 --json
t $P -m trainsuite Training.status nope-xyz
t $P -m runsuite RunJobs.status nope-xyz
t $P -m evalsuite Evaluation.quant nope@e123
t $P -m evalsuite Runner.run runs/nope-xyz val
t $P -m evalsuite Runner.check runs/nope-xyz val
t $P -m evalsuite Evaluation.check runs/nope-xyz val
t $P -m trainsuite Training.check /nonexistent.json
t $P -m evalsuite nonsense
