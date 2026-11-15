cd "$(dirname "$0")/../.."
for f in temp/e306/start_exact.sh temp/e298/start_driven.sh temp/e302/start_cap.sh; do
  [ -f $f.orig ] || cp $f $f.orig
  cat > $f <<'STUB'
#!/bin/bash
# skipped on 2026-11-12 10:45 (day chain 4 slot freed for the combined maps arm): the pair decision is at the level of a perfect classifier on the snapped read-out (E310 / E311); the original is the .orig file next to this one
echo "skipped: $0 $* ($(date))"
exit 0
STUB
done
ls -la temp/e306 temp/e298 temp/e302 | head -20
