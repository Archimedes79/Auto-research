#!/bin/bash
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
bash tools/env/_py14.sh e309/lines_stats.py 0 > temp/e309/lines_stats_run.log 2>&1
bash tools/env/_py14.sh e309/offsets.py 0 > temp/e309/offsets_run.log 2>&1
bash tools/env/_py14.sh e309/scenes.py > temp/e309/scenes_run2.log 2>&1
for S in 0 1; do bash tools/env/_py14.sh jends_eval.py score --run e312_p_s$S --seed $S --thr 0.1 > temp/e309/e312_unsnap_s$S.log 2>&1; done
echo JSON DONE > temp/e309/run_json.done
