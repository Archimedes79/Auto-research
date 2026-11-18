#!/bin/bash
# the lead's 06:50 check: runs/e372_s4_s0 read LIVE through the landed code (2ade9bd41d) reproduces its stored live card bit for bit.
# Waits for the day window (08:30), then one GPU live dump of all 873 VAL frames as a NEW file (--max 873 -> token_val_live_n873.pkl; the stored dump is never
# overwritten), its CPU score at D115 (cards to temp/e365/out), and a comparison of every metric with the stored card. Diagnostic (run.py --no-card).
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
L=temp/e385/repro_s4.log
say() { echo "$(date '+%m-%d %H:%M:%S') [repro_s4] $*" >> $L; }
say "waiting for 08:30"
until [ "$(date +%H%M)" -ge 0830 ] && [ "$(date +%H%M)" -lt 1530 ]; do sleep 30; done
D115="--link-thr 0.3 --link-thr-long 0.2 --keep-thr 0.6 --no-none-gate"
python3 ../autoresearch/tools/run.py --id E372.repro_s4_s0 --seed 0 --gpu --retries 1 --no-card --split val --timeout 45 --hyp "the landed code reads e372_s4_s0 live as stored" --pred "bit for bit" --rule "diagnostic: every metric equal to the stored live card" -- bash -c "bash tools/env/_pyr.sh jtoken_eval.py dump --live --run e372_s4_s0 --seed 0 --split val --max 873 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh jtoken_eval.py score --live --run e372_s4_s0 --seed 0 --max 873 $D115" >> temp/e385/repro_s4.out 2>&1
say "run exit $?"
python3 - >> $L 2>&1 <<'EOF'
import glob, json
old = json.load(open("runs/e372_s4_s0/token_card_rec_lt0.3_ltl0.2_kt0.6_nng_live_vs_e373_cb_s0.json"))["metrics"]
new_p = sorted(glob.glob("temp/e365/out/e372_s4_s0_token_card_rec_lt0.3_ltl0.2_kt0.6_nng_live_n873*.json"))
new = json.load(open(new_p[-1]))["metrics"]
keys = sorted(set(old) & set(new))
diff = [k for k in keys if old[k] != new[k] and not (old[k] != old[k] and new[k] != new[k])]
print(f"compare {new_p[-1]} with the stored card: {len(keys)} metrics, differing {len(diff)}: {diff[:10]}; topo_F1 {old['topo_F1']} / {new['topo_F1']}, junction_F1 {old['junction_F1']} / {new['junction_F1']}; {'BIT-FOR-BIT' if not diff else 'DIFFERS'}")
EOF
say "done"
