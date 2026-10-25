#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# copy of everything loop 18 wrote or changed (rollback hazard: files made with Write / Edit can disappear with a conversation rollback; a shell copy survives)
R=$BFLC
B=$R/experiment_execution/loop18_backup
mkdir -p $B
cd $R
for f in $(find training evaluation tools reports/build temp/launchers/loop18 temp/analysis/loop18 experiment_execution/indexes/LOOP18_STATE.md HANDOVER.md -type f \( -name "*.py" -o -name "*.sh" -o -name "*.md" \) ! -name README.md -newermt "2026-10-24 22:30" 2>/dev/null | grep -v __pycache__); do
  mkdir -p "$B/$(dirname $f)"; cp -p "$f" "$B/$f"
done
find $B/training $B/evaluation $B/tools $B/reports $B/temp -name "*.md" -exec mv {} {}.txt \; 2>/dev/null
echo "backup $(date '+%F %H:%M'): $(find $B -type f | wc -l) files" >> $B/BACKUP_LOG.txt
tail -1 $B/BACKUP_LOG.txt
