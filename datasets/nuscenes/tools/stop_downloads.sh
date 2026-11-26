#!/usr/bin/env bash
# Stop the running nuScenes archive downloads (wrapper scripts first, so their fallback curl does not restart,
# then the curl processes). Partial archives are kept; get_blob_part.sh resumes them with curl -C -.
pkill -f "[g]et_parts_parallel.sh" || true
pkill -f "[g]et_blob_part.sh" || true
sleep 1
pkill -f "[m]otional-nuscenes" || true
sleep 2
ps -eo pid,cmd | grep -E "[c]url|[g]et_parts|[g]et_blob" || echo "no download running"
