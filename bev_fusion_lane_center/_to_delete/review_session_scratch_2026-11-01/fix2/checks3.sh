cd /mnt/c/Auto-research/bev_fusion_lane_center
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
nice python3 ../autoresearch/tools/selftest.py > /tmp/ar_st_dir2.txt 2>&1; echo "autoresearch selftest exit $?"
grep -v '^note: S.c' /tmp/ar_st_dir2.txt | grep -B12 -A3 'Traceback' | head -30
python3 /mnt/c/Users/mmeut/AppData/Local/Temp/claude/C--Auto-research/6e9638e0-a42f-4660-b7d5-cac910d9084d/scratchpad/recheck_decisions.py
bash tools/env/_py14.sh evaluate.py --golden check --cached-only 2>&1 | tail -1
python3 ../autoresearch/tools/card.py check --all 2>&1 | grep -v "^ok" | tail -5
python3 ../autoresearch/tools/experiment.py check --all 2>&1 | grep -A3 "NOT READY"
python3 -c "import json; c = json.load(open('runs/eval/fast_recall/val_golden_check.json')); print('better entries:', len(c['better']), 'lower:', sorted(k for k, v in c['better'].items() if v == 'lower'))"
bash tools/env/_py14.sh sparse_lanes/selftest.py 2>&1 | tail -1
