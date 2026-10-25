#!/bin/bash
# E134 sanity: the untrained gate reproduces the spatial hold
S=/mnt/c/ML-Experiment/scripts
bash $S/_py14.sh p17_gate.py apply --init_only --perts dark --out gate0
bash $S/_py14.sh p17_gate_check.py
