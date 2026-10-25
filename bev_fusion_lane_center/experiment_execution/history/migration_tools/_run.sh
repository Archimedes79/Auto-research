#!/bin/bash
# run a python file in the gcar env without changing into the scripts dir (the dir is moved by the migration)
cd /mnt/c/ML-Experiment/restructure
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gcar
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
python "$@"
