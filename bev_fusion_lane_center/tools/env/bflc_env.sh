# source this after "conda activate gcar": module search path of the project (flat imports: import graph_eval as GE ...)
R=/mnt/c/ML-Experiment/bev_fusion_lane_center
_P=""
for _d in $(find $R/training $R/evaluation $R/tools $R/reports/build $R/temp -maxdepth 4 -type d ! -name __pycache__ 2>/dev/null); do _P="$_P:$_d"; done
export PYTHONPATH="${_P#:}:${PYTHONPATH:-}"
