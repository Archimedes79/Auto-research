#!/bin/bash
# scratch: the help of trainsuite queue add / serve
cd /mnt/c/Auto-research/autoresearch || exit 1
export AR_PROJECT=../bev_fusion
python3 -m trainsuite queue add --help 2>&1 | head -60 | cut -c1-240
echo ---serve
python3 -m trainsuite queue serve --help 2>&1 | head -30 | cut -c1-240
echo ---queue dir
ls -la /mnt/c/Auto-research/bev_fusion/records/queue 2>&1 | head; ls /mnt/c/Auto-research/bev_fusion/records | head -20
