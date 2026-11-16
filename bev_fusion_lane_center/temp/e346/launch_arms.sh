#!/bin/bash
# E346: submits the four arms (the --after chain of temp/e346/start_arms.txt makes them run one after the other); logs temp/e346/launch.log. usage: wsl bash temp/e346/launch_arms.sh
# the loop runs in the MAIN shell (process substitution, not a pipe) so that `wait` waits for the launchers: a pipe made them children of a subshell and they died with the script
cd "$(dirname "$0")/../.."
L=temp/e346/launch.log
echo "launch $(date)" >> $L
while IFS= read -r line; do
  bash -c "$line" >> temp/e346/arms_stdout.log 2>&1 &
  sleep 3
done < <(grep -E '^env CUDA_VISIBLE_DEVICES' temp/e346/start_arms.txt)
wait
echo "all four launchers ended $(date)" >> $L
