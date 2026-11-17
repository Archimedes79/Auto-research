#!/bin/bash
# E383 supervisor (one family on one GPU slot): waits for the night window (22:00), then
#   0. the tests at the launch state (CPU, then the GPU tests of the speed path)
#   1. the launch-gate benchmark (run.py with the card): the speed path must be >= 1.3 x the plain step, peak < 4 GB, or stop
#   2. the K 384 caches into their own root runs/e383_cache (seed 0 / 1 x val / train), then the K 512 / K 384 step-time pair (ABBA, one process)
#   3. f_s0, k_s0, f_s1, k_s1 (jtoken_fast_train.py, resume after a crash, stop on a non-finite loss), each followed by its dumps and CPU read-out; f's loss curve is checked against the E373 head's (a kill band)
# usage: bash temp/e383/sup383.sh      (HOLD: touch temp/e383/HOLD stops it before the next stage; no new stage between 03:00 and 22:00)
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
L=temp/e383/chain383.log
say() { echo "$(date '+%m-%d %H:%M:%S') [sup383] $*" >> $L; }
source temp/e383/texts383.sh
exec 8>/tmp/e383_sup.lock
flock -n 8 || { say "supervisor already runs"; exit 0; }
RUN="python3 ../autoresearch/tools/run.py"
held() {
  [ -e temp/e383/HOLD ] && { say "HOLD file present: stopping before the next stage"; exit 0; }
  h=$(date +%H%M); [ "$h" -ge 0300 ] && [ "$h" -lt 2200 ] && { say "after 03:00: no new stage (the quiet-host cost job of the lane chat is armed from 04:30; the weekday window is 08:30): stopping, restart the supervisor in the next night window"; exit 0; }
}
say "waiting for the night window (22:00)"
until [ "$(date +%H%M)" -ge 2200 ] || [ "$(date +%H%M)" -lt 0500 ]; do sleep 30; done
say "night window open"
held
# the lead's condition (22:00 slot word): both critic passes ACCEPT (card with G2 yes, code) before anything starts: the report chat creates temp/e383/CRITICS_OK only then
if [ ! -e temp/e383/CRITICS_OK ]; then say "waiting for temp/e383/CRITICS_OK (both critic passes ACCEPT)"; fi
until [ -e temp/e383/CRITICS_OK ]; do held; sleep 60; done
say "critics ok"
# ---- 0. the tests at the launch state
if [ ! -e temp/e383/tests_passed ]; then
  say "tests at launch"
  ( bash tools/env/_py14.sh training/grid/test_jtoken_fast.py && bash tools/env/_py14.sh training/grid/test_lane_ptok_gpu.py && bash tools/env/_py14.sh training/grid/test_build_tokcache_root.py && bash tools/env/_py14.sh training/grid/test_jtoken.py ) > temp/e383/tests_cpu.log 2>&1 || { say "CPU tests FAILED: stopping (temp/e383/tests_cpu.log)"; exit 1; }
  $RUN --id E383.gputests --gpu --ram-mb 3000 --retries 0 --require-card --timeout 10 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- env JTF_GPU=1 bash tools/env/_py14.sh training/grid/test_jtoken_fast.py gpu >> temp/e383/tests_gpu.log 2>&1
  grep -q "gpu ok" temp/e383/tests_gpu.log || { say "GPU tests FAILED: stopping"; exit 1; }
  touch temp/e383/tests_passed
  say "tests passed"
fi
held
# ---- 1. the benchmark (launch gate)
if [ ! -e temp/e383/gate_passed ]; then
  say "benchmark (launch gate)"
  $RUN --id E383.bench_gate --gpu --ram-mb 6000 --retries 0 --require-card --timeout 20 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash tools/env/_py14.sh /mnt/c/Auto-research/bev_fusion_lane_center/temp/_session_scratch_prof/bench_fast.py gate >> temp/e383/bench.out 2>&1
  python3 - <<'PYEOF' || { say "gate FAILED or no benchmark result: stopping (the card's launch gate)"; exit 1; }
import json, sys
r = json.load(open("temp/e383/bench_gate.json"))
plain = r["plain_cpu_loader_w2"]["ms_per_step"]
fast = min(r["fast_head_gpu_loader"]["ms_per_step"], r["fast_head_gpu_loader_compile"]["ms_per_step"])
peak = max(r[k]["peak_gb"] for k in ("fast_head_gpu_loader", "fast_head_gpu_loader_compile"))
print("plain", plain, "fast", fast, "ratio", round(plain / fast, 2), "peak GB", peak)
open("temp/e383/gate.txt", "w").write(f"plain {plain} ms, fast {fast} ms (best of eager / compile), ratio {plain / fast:.2f}, peak {peak} GB, host load {r['host_load']:.1f}\n")
sys.exit(0 if plain / fast >= 1.3 and peak < 4.0 else 1)
PYEOF
  touch temp/e383/gate_passed
  say "gate passed: $(cat temp/e383/gate.txt)"
fi
held
# ---- 2. the K 384 caches (their own root: a second K next to the K 512 caches would break every call without --K)
for S in 0 1; do for SP in val train; do
  if ls runs/e383_cache/be292_a_${S}_K384_*/$SP/manifest.json > /dev/null 2>&1; then continue; fi
  held
  say "cache build seed $S $SP K 384"
  $RUN --id E383.tc_a_s${S}_${SP} --seed $S --gpu --ram-mb 6000 --retries 1 --require-card --split val --timeout 30 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash tools/env/_pyr.sh build_tokcache_root.py --root runs/e383_cache --run runs/e292_a_s$S --split $SP --K 384 --no-f2 --workers 2 >> temp/e383/tc_${S}_${SP}.out 2>&1
  say "cache seed $S $SP exit $?"
done; done
if [ ! -e temp/e383/bench_pair.json ]; then
  held
  say "step-time pair K 512 / K 384 (ABBA)"
  $RUN --id E383.bench_pair --gpu --ram-mb 6000 --retries 0 --require-card --timeout 20 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash tools/env/_py14.sh /mnt/c/Auto-research/bev_fusion_lane_center/temp/_session_scratch_prof/bench_fast.py pair >> temp/e383/bench.out 2>&1
  say "pair benchmark exit $?: $(grep -E 'ratio_384' temp/e383/bench_pair.json 2>/dev/null | tr -d ' ')"
fi
# ---- 3. the arms
curve_ok() {   # f's loss curve against the E373 head of the same cache and head seed (the card's kill band): |diff| <= 0.05 at step 500, 0.03 at 2000, 0.02 at 12000
  python3 - "$1" <<'PYEOF'
import json, sys
S = sys.argv[1]
def curve(p):
    return {r["step"]: r["total"] for r in (json.loads(l) for l in open(p)) if "total" in r}
a, b = curve(f"runs/e383_f_s{S}/log.jsonl"), curve(f"runs/e373_cb_s{S}/log.jsonl")
band = {500: 0.05, 2000: 0.03, 12000: 0.02}
bad = {k: (round(a[k], 4), round(b[k], 4)) for k, w in band.items() if abs(a[k] - b[k]) > w}
print("curve f vs E373:", {k: (round(a[k], 4), round(b[k], 4)) for k in band}, "BAD" if bad else "ok")
sys.exit(1 if bad else 0)
PYEOF
}
train_arm() {   # arm (f|k), S, H
  local arm=$1 S=$2 H=$3 NAME=e383_${1}_s$2 KARG="" n=0
  [ "$arm" = k ] && KARG="--cache runs/e383_cache --K 384"
  while ! grep -q "^[0-9:]* done $NAME\$" runs/$NAME/log.txt 2>/dev/null; do
    held
    if pgrep -f "run.py --id E383.${arm}_s$S " > /dev/null; then sleep 30; continue; fi
    if grep -qiE "loss (nan|inf)|not finite" runs/$NAME/log.txt temp/e383/train_${arm}_s$S.out 2>/dev/null; then say "${arm}_s$S: a non-finite loss (the card's kill): no relaunch"; return 1; fi
    n=$((n+1)); [ $n -gt 8 ] && { say "${arm}_s$S: 8 launches, giving up"; return 1; }
    say "${arm}_s$S launch $n (cache seed $S, head seed $H, ${KARG:-K 512})"
    $RUN --id E383.${arm}_s$S --seed $S --gpu --ram-mb 6000 --retries 0 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash tools/env/_pyr.sh jtoken_fast_train.py --name $NAME --seed $S --head-seed $H --ident e292_a --steps 12000 --bs 8 --lr 1e-3 --workers 4 --none-weight 0.25 $KARG >> temp/e383/train_${arm}_s$S.out 2>&1
    sleep 20
  done
  say "${arm}_s$S training done"
  if [ "$arm" = f ]; then out=$(curve_ok $S); say "$out"; case "$out" in *BAD*) say "f_s$S: the loss curve is outside the card's band (kill): no further arm"; return 1;; esac; fi
  bash temp/e383/read383.sh dump ${arm}_s$S >> temp/e383/read_${arm}_s$S.out 2>&1
  bash temp/e383/read383.sh cpu ${arm}_s$S >> temp/e383/read_${arm}_s$S.out 2>&1
  say "${arm}_s$S read-out done"
}
train_arm f 0 2 || exit 1
train_arm k 0 2
train_arm f 1 3 || exit 1
train_arm k 1 3
say "all arms done"
