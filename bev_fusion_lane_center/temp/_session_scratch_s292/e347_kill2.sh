#!/bin/bash
# E347 kill criterion 2 (card section 2b (c)): wait until arm <A> seed <S> logged step 500, then score its ckpt.pt (CPU, nice, one scorer at a time via flock)
# usage: e347_kill2.sh g|o 0|1
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
A=$1; S=$2; R=e347_${A}_s$S
n=0
until grep -qE "step (5[0-9][0-9]|[6-9][0-9][0-9]|[1-9][0-9]{3})/3000" runs/$R/train.log 2>/dev/null; do
  n=$((n+10)); [ $n -gt 3600 ] && { echo "no step 500 after 1 h"; exit 2; }
  sleep 10
done
echo "$R reached step 500 $(date +%H:%M:%S): $(grep -E 'step 500/3000' runs/$R/train.log | cut -c1-160)"
exec 8>/tmp/e347_kill2.lock
flock 8
python3 ../autoresearch/tools/run.py --id E347.kill2_${A}_s$S --seed $S --retries 0 --no-card --split val --timeout 60 \
  --hyp "E347 kill criterion 2 (diagnostic): the checkpoint of $R after 500 steps has keypoints above 0.3 on the first VAL frames" \
  --pred "pred > 0" --rule "by hand: kill gate of the card (zero predictions at 0.3 = kill; read gate_sweep before reading a zero on arm o)" \
  -- bash temp/e347/run_staged.sh jmaps_eval.py score --run $R --ckpt --max 50 --workers 0
echo "kill2 $R exit $? $(date +%H:%M:%S)"
python3 - <<PYEOF
import json
d = json.load(open("runs/$R/jmaps_val_ckpt.json"))
k = d["keypoints"]
print("step", d.get("step"), "frames", d.get("frames"), "ours_tol3.0_ent.pred", k.get("kp_ours_tol3.0_noang_ent.pred"), "ours_tol3.0_ext.pred", k.get("kp_ours_tol3.0_noang_ext.pred"))
gs = d.get("gate_sweep") or d.get("gate")
print("gate_sweep", json.dumps(gs)[:600])
PYEOF
