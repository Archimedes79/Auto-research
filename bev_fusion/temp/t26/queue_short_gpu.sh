#!/bin/bash
# t26: the short GPU jobs AFTER the Spurmitte lane work (Mirko, 2026-11-15: "Run them after spurmitte"): wait until the GPU is quiet (no other run.py --gpu job, GPU memory < 2.5 GB, utilisation < 10 %, 5 looks one minute apart),
# then CRN smoke -> CRN full export -> the t26 training smoke (that one needs the GPU alone: 10 quiet looks first), strictly one after another, every one through run.py (the Training class for the smoke).
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=../bev_fusion
log() { echo "== $(date '+%F %T') $*"; }

quiet() {   # $1 looks in a row, $2 seconds between the looks
  local ok=0 mem util other
  while [ $ok -lt $1 ]; do
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
    other=$(ps -eo args | grep -E "tools/run.py --id" | grep -e "--gpu" | grep -v -E "t26\.|smoke\.t26|grep" | wc -l)
    if [ "${mem:-99999}" -lt 2500 ] && [ "${util:-100}" -lt 10 ] && [ "$other" -eq 0 ]; then ok=$((ok+1)); else ok=0; fi
    sleep $2
  done
}

R="python3 ../autoresearch/tools/run.py --seed 0 --retries 0 --require-card --split val --gpu --gpu-max-mb 3000"

log "waiting for a quiet GPU (2 looks)"; quiet 2 15
log "start t26.ref_crn_smoke"
$R --id t26.ref_crn_smoke --hyp "CRN R50 runs on our frames on the RTX 3060 within about 5 GB; TF32 off against on moves box centres by at most 0.1 m on 50 samples; the converted boxes land on our ground truth." --pred "peak memory 3 - 5 GB, ms per sample below 400, largest matched-centre difference TF32 on / off <= 0.1 m, overlay hits >= 90 of 100 GT boxes" --rule "gate b (centres within 0.1 m, no box lost) and gate a (the overlay by eye) of card t26 section 1; a failure makes the object row maps only; no other verdict" -- bash temp/t26/job_crn_smoke.sh
log "t26.ref_crn_smoke rc=$?"

log "waiting for a quiet GPU (2 looks)"; quiet 2 15
log "start t26.ref_crn"
$R --id t26.ref_crn --hyp "CRN R50 scores a devkit mAP within 0.473 +- 0.05 on the 1706 samples of the 43 common scenes, and its boxes converted to our frames give the reference for the object row." --pred "devkit mAP 0.42 - 0.52 (gate c, light tables); the front-sector mAP2 of CRN on the 1620 frames above the base's" --rule "gate c of card t26 section 1 (else maps only); the object row itself is composed by hand from the table rows" -- bash temp/t26/job_crn_full.sh
log "t26.ref_crn rc=$?"

log "waiting for a quiet GPU and 10 GB of free memory for the timing smoke"; quiet 3 20; ok=0; while [ $ok -lt 3 ]; do a=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo); if [ "$a" -ge 10240 ]; then ok=$((ok+1)); else ok=0; fi; sleep 30; done
log "start smoke.t26_one_s0 (Training run)"
cd /mnt/c/Auto-research/autoresearch || exit 1
export PYTHONPATH=/mnt/c/Auto-research:/mnt/c/Auto-research/autoresearch
~/miniconda3/envs/gcar/bin/python -m Training run ../bev_fusion/temp/t26/job_smoke.json 2>&1 | tail -5 | cut -c1-300
cd /mnt/c/Auto-research/bev_fusion || exit 1
until [ -e records/runs/smoke.t26_one_s0/done ]; do
  s=$(python3 -c 'import json;print(json.load(open("records/runs/smoke.t26_one_s0/run.json")).get("status"))' 2>/dev/null)
  case "$s" in failed|blocked|stalled|stopped|held) break;; esac
  sleep 60
done
log "smoke.t26_one_s0 status=$s"
log "all short jobs done"
