#!/bin/bash
# scratch: back up the best network (keep3 slot 1: t21_ctl_s0, the evaluated checkpoint) to D: and verify it against the sha256 of the record
set -e
SRC=/mnt/c/Auto-research/bev_fusion/runs/t21_ctl_s0
DST=/mnt/d/Auto-research/backup_best_network_2026-11-10/t21_ctl_s0
mkdir -p "$DST"
cp -r "$SRC/code" "$DST/code"
cp "$SRC/config.json" "$SRC/gt.json" "$SRC/log.jsonl" "$SRC/train.log" "$DST/"
mkdir -p "$DST/eval" && cp "$SRC/eval/metrics.json" "$SRC/eval/card.json" "$DST/eval/" 2>/dev/null || cp "$SRC/eval/metrics.json" "$DST/eval/"
cp "$SRC/ckpt_00001500.pt" "$DST/"
want=$(python3 -c "
import json
k = json.load(open('/mnt/c/Auto-research/bev_fusion/records/keep3.json'))
for e in k['entries']:
    for c in e['checkpoints']:
        if c['path'].endswith('t21_ctl_s0/ckpt_00001500.pt'): print(c['sha256'])
")
got=$(sha256sum "$DST/ckpt_00001500.pt" | cut -d' ' -f1)
echo "record sha256: $want"; echo "copy   sha256: $got"
[ "$want" = "$got" ] && echo "BACKUP VERIFIED" || { echo "MISMATCH"; exit 1; }
cat > "$DST/../README.txt" <<'EOF'
Backup of the best network of 2026-11-10 (Mirko: "Sichere das beste Netzwerk, loesche alles"): bev_fusion/runs/t21_ctl_s0 (keep3 slot 1, board7 0.4549 on GT v8 0.5_4_6a7aa7864a, val):
the evaluated checkpoint ckpt_00001500.pt (sha256 as in bev_fusion/records/keep3.json), its config, GT record, code snapshot, training log and evaluation card.
Restore: copy the folder back to bev_fusion/runs/t21_ctl_s0.
EOF
du -sh "$DST"
