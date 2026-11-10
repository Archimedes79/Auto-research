cd "$(git rev-parse --show-toplevel)"
PYTHONPATH=$(pwd) $HOME/miniconda3/envs/gcar/bin/python -c "import autogt; print('autogt', autogt.__file__)" 2>&1 | tail -1
PYTHONPATH=$(pwd) $HOME/miniconda3/envs/gcar/bin/python -c "import GtGenerator; print('GtGenerator', GtGenerator.__file__)" 2>&1 | tail -1
grep -n "autogt" bev_fusion/runs/t20_base_s0/code/*.py | head -3 | cut -c1-140
