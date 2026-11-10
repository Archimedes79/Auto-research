# usage: wsl bash bev_fusion/temp/pitch3/py.sh <script.py> [args]   (from the repository root)
cd "$(git rev-parse --show-toplevel)"
exec $HOME/miniconda3/envs/gcar/bin/python "$@"
