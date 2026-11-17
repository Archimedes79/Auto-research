#!/usr/bin/env bash
# export the winner folder of the E2E token champion (D114): source + tests, repo-relative paths, no weights
cd /mnt/c/Auto-research || exit 1
PY=/home/ml/miniconda3/envs/gcar/bin/python
B=/mnt/c/Auto-research/bev_fusion_lane_center
$PY reporter/winner.py export e365_token_champion \
  --src "$B/training/grid/jtoken.py" --src "$B/training/train/jtoken_train.py" --src "$B/training/data/lane_ptok_data.py" --src "$B/training/data/lane_tokcache.py" \
  --src "$B/evaluation/jtoken_eval.py" --src "$B/training/grid/test_jtoken.py" --src "$B/training/grid/test_lane_tokcache.py" --src "$B/evaluation/tests/test_jtoken_eval.py" \
  --strip "$B" \
  --title "E2E token champion (D114): one token head on the predicted nodes of the frozen lane head B' (source + tests, no weights)" \
  --numbers "VAL 873 frames, four heads at the decode chosen on the TRAIN sample (link p >= 0.1, node p_none < 0.6, none gate off): topo F1 0.6617 / 0.6767 (head seeds 0 / 1, E366) and 0.6618 / 0.6733 (fresh head seeds 2 / 3, E371), floors 0.6359 / 0.6349, plain lane head B 0.6259 / 0.6249; F1@1.0 0.836 / 0.842 / 0.840 / 0.840; lanes found 0.675 / 0.691 / 0.685 / 0.685. Whole path 16.8 / 16.6 ms (quiet window, 24 frames, batch 1, bf16) against 19.7 ms for E324 s; 2.323 M parameters (B' 2.196 M + head 0.127 M). Costs: precision 0.69, dup_frac 0.055 - 0.063, 25 vertices per 100 m, junction F1 0.20 - 0.21 (below the plain head's 0.27). No TEST number. Status: stage S1, B' frozen." \
  --command "NEEDS the repository (it imports jtok, jtok_eval, qgraph_eval, qgraph_readout, qtargets, trainer): copy code/ over bev_fusion_lane_center/ and run from there. Train: bash tools/env/_py14.sh jtoken_train.py --name e365_s1_s<S> --seed S --ident e292_a --steps 12000 --bs 8 --lr 1e-3 --workers 4. Read out: bash tools/env/_py14.sh jtoken_eval.py dump --run e365_s1_s<S> --seed S; jtoken_eval.py score --run e365_s1_s<S> --seed S --link-thr 0.1 --keep-thr 0.6 --no-none-gate. Time: jtoken_eval.py timing --run e365_s1_s0 --b-run runs/e292_a_s0. Tests: bash tools/env/_py14.sh training/grid/test_jtoken.py (and the two other test files)." \
  --weights "head: runs/e365_s1_s0, runs/e365_s1_s1, runs/e371_cf_s0 (head seed 2), runs/e371_cf_s1 (head seed 3); frozen lane head B': runs/e292_a_s0 / s1; token cache runs/e365_cache/be292_a_{0,1}_K512_9ad1d4ce30_v1 (derived). All in the repository runs, not in git."
$PY reporter/winner.py verify winners/e365_token_champion
du -sb winners/e365_token_champion/code | cut -f1
