#!/bin/bash
# E354 cost line (D86 / D106): ms per frame of the junction system with the recursive state carried, on a QUIET host, as the lane lead's protocol says:
# start NO GPU or loaded CPU job until the flag temp/_session_scratch_s292/quiet_window_done exists (the lane chat's E351 / E353 timing watcher writes it when its parts have run);
# then a smoke of the GPU path (one repeat, 8 frames, no --force: the tool itself refuses a loaded host), then the real run (e341_p_s0, e354_z_s0, e354_r_s0, one scene of 40 frames, 4 repeats).
# usage: time_chain.sh   (waits until 07:10 at the latest; every job through run.py). Never edit this script while it runs.
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
FLAG=temp/_session_scratch_s292/quiet_window_done
L=temp/e354/time_chain.log
RUN="python3 ../autoresearch/tools/run.py"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $L; }
exec 9>/tmp/e354_time_chain.lock
flock -n 9 || { say "time chain already runs"; exit 0; }
say "waits for $FLAG"
until [ -f $FLAG ]; do
  [ "$(date +%H%M)" -ge 0710 ] && [ "$(date +%H%M)" -lt 1200 ] && { say "no quiet window by 07:10: ms stay open"; exit 1; }
  sleep 10
done
say "flag exists: $(tr '\n' ' ' < $FLAG | cut -c1-200); load $(cut -d' ' -f1-3 /proc/loadavg); GPU $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
TAG=e354_quiet_$(date +%m%d_%H%M)
$RUN --id E354.time_smoke --retries 0 --no-card --split val --gpu --timeout 15 -- bash tools/env/_py14.sh jrecur_time.py time --runs e354_r_s0 --scene-frames 8 --repeats 1 --tag smoke_$TAG >> temp/e354/time_chain_out.log 2>&1
rc=$?; say "smoke exit $rc"
[ $rc -ne 0 ] && { say "the smoke failed: no real run (the GPU path needs a fix)"; exit 1; }
$RUN --id E354.time --seed 0 --retries 0 --require-card --split val --gpu --timeout 40 \
  --hyp "The recursive feature state of E354 (a warp, a gated 3 x 3 fusion, one more tensor in and out) adds a small cost per frame to the junction system: milliseconds on a quiet host at batch 1, the state carried through one scene." \
  --pred "r minus E341 about +1.0 ms (window +0.3 to +2.5) of about 14.7 ms; z minus E341 about +0.3 ms (the fusion convolution runs on a zero state)" \
  --rule "E354 card section 3: a cost line, no verdict; reported with paired intervals against E341 and the empty-state twin; refused on a loaded host (load < 3, GPU idle) as jtiming.py" \
  -- bash tools/env/_py14.sh jrecur_time.py time --runs e341_p_s0,e354_z_s0,e354_r_s0 --scene-frames 40 --repeats 4 --tag $TAG >> temp/e354/time_chain_out.log 2>&1
rc=$?; say "timing exit $rc, tag $TAG"
