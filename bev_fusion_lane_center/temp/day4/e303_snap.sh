cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
S=$1
until [ -f runs/e303_s_s$S/ends_val.pkl ] && [ -f runs/e303_chord_s$S/ends_val.pkl ]; do sleep 20; done
sleep 30
# learned handle scales with snapped ends (p >= 0.1 / 0.5) against the chord tangent with snapped ends, same pairs
bash tools/env/_py14.sh jends_eval.py score --run e303_s_s$S --seed $S --snap --thr 0.1 >> temp/day4/score_e303_snap_s$S.log 2>&1
bash tools/env/_py14.sh jends_eval.py score --run e303_s_s$S --seed $S --snap --thr 0.5 >> temp/day4/score_e303_snap_s$S.log 2>&1
bash tools/env/_py14.sh jends_eval.py score --run e303_chord_s$S --seed $S --chord --snap --thr 0.1 >> temp/day4/score_e303_snap_s$S.log 2>&1
bash tools/env/_py14.sh jends_eval.py score --run e303_chord_s$S --seed $S --chord --snap --thr 0.5 >> temp/day4/score_e303_snap_s$S.log 2>&1
echo "E303 snapped scored s$S $(date)" >> temp/day4/redo.log
