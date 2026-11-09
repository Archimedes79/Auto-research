#!/bin/bash
# Mirko 2026-11-03: "Yes, delete them" = the ledger rows and the run records (planned runs, no data, no checkpoints) of t16.ctl_s0, t16.ctl_s0.eval, speed.d, speed.e, speed.f.
# A backup of what goes is kept in temp/speed_tools/deleted_held_runs_2026-11-03/ (scratch, not in git).
cd /mnt/c/Auto-research/bev_fusion || exit 1
IDS="t16.ctl_s0 t16.ctl_s0.eval speed.d speed.e speed.f"
B=temp/speed_tools/deleted_held_runs_2026-11-03
mkdir -p "$B"
for id in $IDS; do
  d=records/runs/$id
  [ -d "$d" ] || { echo "no record dir $d"; continue; }
  # only the three small files may be in there, else stop
  extra=$(ls -A "$d" | grep -v -x -e HOLD -e heartbeat -e run.json)
  [ -z "$extra" ] || { echo "STOP: $d holds more: $extra"; exit 1; }
  mkdir -p "$B/$id" && cp -p "$d"/* "$d"/HOLD "$B/$id/" 2>/dev/null
  grep -F "$id," records/LEDGER.csv | grep "^$id," >> "$B/ledger_rows.csv"
done
echo "backup rows: $(wc -l < $B/ledger_rows.csv)"
python3 - <<'EOF'
ids = "t16.ctl_s0 t16.ctl_s0.eval speed.d speed.e speed.f".split()
p = "records/LEDGER.csv"
raw = open(p, encoding="utf-8", newline="").read()
lines = raw.split("\n")
keep = [ln for ln in lines if not any(ln.startswith(i + ",") for i in ids)]
print("removed", len(lines) - len(keep), "lines")
open(p, "w", encoding="utf-8", newline="").write("\n".join(keep))
EOF
for id in $IDS; do rm -r "records/runs/$id"; done
git diff --stat -- records/LEDGER.csv records/LEDGER.md
ls records/runs | grep -E "^(t16.ctl|speed\.[def]$)" || echo "run records gone"
