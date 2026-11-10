#!/bin/bash
# scratch: do the lane structure problems mention anything I removed in the documentation cleanup?
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
python3 ../autoresearch/tools/check_structure.py > /tmp/cs_lane.txt 2>&1
tail -n 1 /tmp/cs_lane.txt
grep -c . /tmp/cs_lane.txt
echo "--- mentions of removed things:"
grep -i -E "migration_tools|literature_sources|SCHEMA" /tmp/cs_lane.txt | head
echo "--- by file (top):"
cut -d: -f1 /tmp/cs_lane.txt | sort | uniq -c | sort -rn | head -8
