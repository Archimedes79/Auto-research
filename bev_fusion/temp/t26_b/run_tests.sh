#!/usr/bin/env bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONPATH=/mnt/c/Auto-research
PY=~/miniconda3/envs/gcar/bin/python
run() { local name="$1"; shift; local t=$(date +%s); if out=$(nice -n 15 "$@" 2>&1); then echo "PASS  $name ($(( $(date +%s) - t )) s) | $(echo "$out" | grep -v -i warn | tail -1 | cut -c1-150)"; else echo "FAIL  $name | $(echo "$out" | grep -v -i warn | tail -6 | tr '\n' ' ' | cut -c1-400)"; fi; }
run metrics            $PY evaluation/metrics.py
run front_compare      $PY evaluation/front_compare.py
run bev_junction       $PY evaluation/bev_junction.py
run soa_selftest       $PY evaluation/soa_maps.py --selftest
run soa_mutants        $PY evaluation/soa_maps.py --mutants
run bev_selftest       $PY evaluation/bev_selftest.py
run evaluate_door      $PY tests/test_evaluate_door.py
run evaluation_api     $PY tests/test_evaluation_api.py
run centreline_eval    $PY tests/test_centreline_eval.py
run bev_errors         $PY evaluation/bev_errors.py --selftest
run bev_response       $PY evaluation/bev_response.py --selftest
run bev_gtcheck        $PY evaluation/bev_gtcheck.py --selftest
run fusion_area        $PY evaluation/bev_fusion_area.py --selftest
run sets               $PY evaluation/bev_sets.py --selftest
echo ALLDONE
