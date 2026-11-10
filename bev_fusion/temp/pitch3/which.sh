PY=$HOME/miniconda3/envs/gcar/bin/python
$PY -m pyflakes --version 2>&1 | head -1
$PY -m vulture --version 2>&1 | head -1
$PY -m flake8 --version 2>&1 | head -1
$PY -m pylint --version 2>&1 | head -1
