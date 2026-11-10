# quick gates of the evaluation area (CPU, no data, no GPU); run from anywhere inside the repository: wsl bash bev_fusion/temp/pitch3/gates.sh
cd "$(git rev-parse --show-toplevel)"
PY=$HOME/miniconda3/envs/gcar/bin/python
export PYTHONPATH="$(pwd)"
row() { name="$1"; shift; if out=$("$@" 2>&1); then echo "PASS $name | $(echo "$out" | grep -v -i warn | tail -1 | cut -c1-150)"; else echo "FAIL $name | $(echo "$out" | grep -v -i warn | tail -4 | tr '\n' ' ' | cut -c1-400)"; fi; }
(cd autoresearch && row "evalsuite selftest" $PY -m evalsuite selftest)
(cd autoresearch && row "evalsuite selftest_api (toy + BEV Runner)" env AR_PROJECT=../bev_fusion $PY -m evalsuite.selftest_api)
(cd autoresearch && row "evalsuite selftest_comparison" $PY -m evalsuite.selftest_comparison)
(cd autoresearch && row "evalsuite selftest_cost" $PY -m evalsuite.selftest_cost)
cd bev_fusion
row "evaluation api (quick)" $PY tests/test_evaluation_api.py
row "suites conform" $PY tests/test_suites_conform.py
row "runner class" $PY tests/test_runner_class.py
row "bev_selftest (synthetic)" $PY evaluation/bev_selftest.py
row "metrics" $PY evaluation/metrics.py
row "front compare" $PY evaluation/front_compare.py
row "curve metrics" $PY evaluation/bev_curves.py --selftest
row "module probe" $PY evaluation/bev_modules.py --selftest
row "module timing" $PY evaluation/bev_timing.py --selftest
row "eval sets" $PY evaluation/bev_sets.py --selftest
row "error search" $PY evaluation/bev_errors.py --selftest
row "fusion area" $PY evaluation/bev_fusion_area.py --selftest
row "evaluate door" $PY tests/test_evaluate_door.py
row "stream tools" $PY tests/test_stream_tools.py
row "centre-line eval" $PY tests/test_centreline_eval.py
row "student chain" $PY tests/test_student_chain.py
row "bev_free import" $PY -c "import evaluation.bev_free, evaluation.bev_probe, evaluation.bev_modules"
echo DONE
