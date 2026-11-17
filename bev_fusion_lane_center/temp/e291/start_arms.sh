#!/bin/bash
# E291 (card experiments/E291_sd-road-level-lane-counts): one pair stage of the E312 recipe through the launcher with the card, on the STAGED code (temp/_session_scratch_e291/stage, found by temp/e291/run_staged_gpu.sh).
#   wsl bash temp/e291/start_arms.sh <c|a|b> <seed>        c = the recipe TWIN (prior flag off: the control of the rule), a = clean counts, b = corrupted counts
# the environment is the one of the E312 launch (experiment_execution/runs/E312.p_s0/run.json: `bash tools/env/_pyr.sh jends_train.py --name e312_p_s0 --seed 0 --maps <runs/e307_n_s0/model.pt> --e2e --scales --maps-w 1.0 --steps 3000 --workers 3`
# and BFLC_SIG=1.0 of its card; BFLC_RASTER defaults to legacy, which matches): BFLC_SIG=1.0 is exported here (the staged jends_train.py stops a run whose BFLC_SIG is not the sigma of the maps checkpoint).
# D101: --workers 4 (the throughput only, the data order is the sampler's); --retries 2 instead of the retry loop of _pyr.sh (run.py repeats a run whose last attempt printed a CUDA 'unknown error'; the trainer resumes from ckpt.pt, with another data order: write "resumed from step N" in the ledger row); --ram-mb 6000 as the E348 script.
# guards (every start): (1) BFLC_STAGE is set HERE, unconditionally, to this stage (E347's wrappers use the same variable name; a shell that exports E347's stage must not start an arm), and run_staged_gpu.sh accepts only that path;
# (2) the freeze: sha256 of the four staged files, jprior.py, evaluation/jdiag.py and temp/e347/stage_launch.py against temp/e291/stage_sha256.txt (refused on a mismatch: the card's code is what was reviewed); (3) a run folder runs/e291_<arm>_s<S> that exists is refused unless E291_RESUME=1
# (a stopped run resumed by hand; run.py's own retries do not pass through this script). Test hooks (no effect on a real start): E291_DRY=1 prints the run.py line instead of starting it, E291_ROOT, E291_SHA_FILE, E291_RUNS_DIR.
cd "${E291_ROOT:-$(dirname "$0")/../..}"
ARM=$1
S=$2
case "$ARM" in c|a|b) ;; *) echo "usage: start_arms.sh <c|a|b> <seed>"; exit 2;; esac
[ -n "$S" ] || { echo "usage: start_arms.sh <c|a|b> <seed>"; exit 2; }
export BFLC_STAGE="$(pwd)/temp/_session_scratch_e291/stage"
unset E291_TEST
SHA="${E291_SHA_FILE:-temp/e291/stage_sha256.txt}"
[ -f "$SHA" ] && sha256sum -c --quiet "$SHA" || { echo "start_arms.sh: a staged file, jprior.py, jdiag.py or stage_launch.py differs from $SHA (the freeze) or the list is missing: REFUSED"; exit 3; }
RUNS="${E291_RUNS_DIR:-runs}"
if [ -e "$RUNS/e291_${ARM}_s$S" ] && [ "$E291_RESUME" != "1" ]; then
  echo "start_arms.sh: $RUNS/e291_${ARM}_s$S exists: REFUSED (a new attempt would mix its log and checkpoint with the old run; E291_RESUME=1 resumes it by hand)"; exit 4
fi
MAPS=$(pwd)/runs/e307_n_s$S/model.pt
[ -f $MAPS ] || { echo "no maps $MAPS"; exit 2; }
export BFLC_SIG=1.0
case "$ARM" in
  c) FLAGS=""
     HYP="The recipe TWIN of E312 (same staged code, environment and data order, prior flag off) reproduces E312 within the noise of the recipe: it is the control of E291 and measures the reproduction noise that no earlier run measured."
     PRED="snapped topo F1 within 0.005 of the stored E312 of the same seed (0.6825 / 0.6864), unsnapped within 0.005 (0.6438 / 0.6481)"
     RULE="E291 card section 0 (by hand: replay, decide.py is not used): the twin is the control of G, R and the shuffled-table kill; if |twin - stored E312| exceeds 0.005 snapped topo F1 on either seed the card says so and decides against the twin; VAL 873, no TEST";;
  a) FLAGS="--lane-prior --prior-noise 0"
     HYP="The number of lanes per direction of a road-level map (arm table of the junction GT, clean counts, 20 % of the junctions without a map) as a gated term on the keypoint tokens of the pair head: what a clean map gives, and what it costs when the map read at test time is wrong (the contrast of a and b at 30 % wrong counts)."
     PRED="snapped topo F1 +0.008 over the twin with the clean table (an oracle on the end targets, an upper bound), -0.02 at 30 % wrong counts"
     RULE="E291 card section 0 (by hand: replay): arm a is reported (table trained at noise 0, read at clean / 10 % / 30 % / +-1 / shuffled); the rule G / R / C is evaluated on arm b; VAL 873, no TEST";;
  b) FLAGS="--lane-prior --prior-noise mix"
     HYP="The same trained with corrupted counts (rate per junction from {0, 0.1, 0.3}, 15 % of the arms unknown, 20 % of the junctions without a map, 20 % with all counts unknown and the geometry kept) learns to doubt the count: it gains with a clean map and does not lose with a wrong one."
     PRED="snapped topo F1 +0.008 over the twin with the clean table, +0.002 with all counts unknown (U), -0.003 at 30 % wrong counts"
     RULE="E291 card section 0 (by hand: replay, decide.py is not used): G / R / C from the pooled paired scene bootstrap of the means over the two seeds against the twin (jdiag.py boot_pooled), clean / 30 % wrong / U read-outs of b; KEEP AS AN OPTIONAL INPUT needs G+, R+, C+; VAL 873, no TEST";;
esac
RUN="python3 ../autoresearch/tools/run.py"
[ "$E291_DRY" = "1" ] && RUN="echo DRY(BFLC_STAGE=$BFLC_STAGE BFLC_SIG=$BFLC_SIG): run.py"
$RUN --id E291.${ARM}_s$S --seed $S --gpu --ram-mb 6000 --retries 2 --require-card --split val \
  --hyp "$HYP" --pred "$PRED" --rule "$RULE" \
  -- bash temp/e291/run_staged_gpu.sh jends_train.py --name e291_${ARM}_s$S --seed $S --maps $MAPS --e2e --scales --maps-w 1.0 --steps 3000 --workers 4 $FLAGS
