#!/bin/bash
# waits for the end of the E320 post read-outs (day12d_post.sh writes temp/day12/DAY12D_DONE), then lists the E320 cards
cd "$(dirname "$0")/../.."
until [ -f temp/day12/DAY12D_DONE ]; do sleep 30; done
echo "DAY12D_DONE $(date)"
ls runs/e320_a_s0/ends_card*.json runs/e320_a_s1/ends_card*.json 2>&1
