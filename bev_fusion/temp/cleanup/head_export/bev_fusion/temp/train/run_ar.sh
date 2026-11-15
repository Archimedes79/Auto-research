#!/usr/bin/env bash
# Redirect (D73, 2026-11-10): the launcher moved to bev_fusion/tools/run_ar.sh (code, with tests/test_run_ar.py). This file stays so that runs recorded with the old command (the held runs,
# the provenance of old runs) still start the same launcher. The next line is not used here: provenance.trainer_python reads the interpreter from the script a run's command names.
PY=~/miniconda3/envs/gcar/bin/python
exec bash "$(dirname "$0")/../../tools/run_ar.sh" "$@"
