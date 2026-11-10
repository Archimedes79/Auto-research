#!/bin/bash
# scratch: are the top arms different checkpoints? (md5 of the evaluated checkpoint) and what the ledger says about them
cd /mnt/c/Auto-research/bev_fusion || exit 1
for r in t21_ctl_s0 t21_ctl_s1 t21_pair4_s0 t21_pair_s0 t21_rec_s0b t22_B_s0 t23_C_s0 t23_C_s1 t20_base_s0 t16_full_s0; do
  f=$(ls runs/$r/ckpt_*.pt 2>/dev/null | tail -1)
  [ -n "$f" ] && echo "$(md5sum "$f" | cut -c1-12)  $(stat -c %s "$f")  $f" || echo "-  -  runs/$r: no checkpoint"
done
echo; for id in t21.ctl_s1 t21.pair4_s0 t21.pair_s0 t21.rec_s0b t22.B_s0 t23.C_s1 t20.base_s0; do
  python3 - "$id" <<'EOF'
import csv, sys
i = sys.argv[1]
for r in csv.DictReader(open("records/LEDGER.csv", encoding="utf-8-sig")):
    if r.get("id") == i:
        print(i, "|", (r.get("hyp") or "")[:90], "|", (r.get("decision") or "")[:80]); break
else:
    print(i, "| no ledger row")
EOF
done
