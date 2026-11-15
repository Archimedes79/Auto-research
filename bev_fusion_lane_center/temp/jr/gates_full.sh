#!/bin/bash
# the fast gates with the full output and the exit code kept (before the push of 2026-11-13)
cd "$(dirname "$0")/../.."
bash tools/maintenance/run_gates.sh > temp/jr/gates_full.out 2>&1
echo "EXIT $?" >> temp/jr/gates_full.out
