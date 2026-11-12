cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
L=temp/day4/redo.log
# the dump of e303_s_s1 died with 'Cannot allocate memory' (host RAM, two dumps beside two trainings): redo it after the dump of seed 0 has ended
until [ -z "$(pgrep -f '[j]ends_eval.py dump --run e303_s_s0')" ]; do sleep 10; done
echo "redo e303 s1 start $(date)" >> $L
bash tools/env/_pyr.sh jends_eval.py dump --run e303_s_s1 >> temp/day4/dump_e303_s_s1_redo.log 2>&1
[ -f runs/e303_s_s1/ends_val.pkl ] || { echo "dump failed again $(date)" >> $L; exit 1; }
bash tools/env/_py14.sh jends_eval.py score --run e303_s_s1 --seed 1 >> temp/day4/score_e303_s_s1_redo.log 2>&1
mkdir -p runs/e303_chord_s1 && cp runs/e303_s_s1/ends_val.pkl runs/e303_s_s1/model.pt runs/e303_chord_s1/
bash tools/env/_py14.sh jends_eval.py score --run e303_chord_s1 --seed 1 --chord >> temp/day4/score_e303_chord_s1_redo.log 2>&1
bash tools/env/_py14.sh jpair_pr.py score --run e303_s_s1 --control e287_p2_s1 >> temp/day4/pr_e303_s_s1_redo.log 2>&1
echo "redo e303 s1 scored $(date)" >> $L
