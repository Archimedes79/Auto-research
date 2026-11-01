#!/usr/bin/env bash
# One-off (2026-10-25): wait (up to 3 h) until no python job of another chat runs (evaluate / run.py / training),
# then build the WSL views with link_views.sh. Run from the repository root.
here="$(cd "$(dirname "$0")/.." && pwd)"
for i in $(seq 1 360); do
  if ! pgrep -f "[p]ython.*(evaluate|run\.py|train|make_|extract)" >/dev/null; then
    echo "$(date +%T) idle, building the views"
    exec bash "$here/link_views.sh"
  fi
  sleep 30
done
echo "$(date +%T) still busy after 3 h, views not built"
exit 1
