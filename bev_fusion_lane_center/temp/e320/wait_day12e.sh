#!/bin/bash
cd "$(dirname "$0")/../.."
until [ -f temp/day12/DAY12E_DONE ]; do sleep 30; done
echo "DAY12E_DONE $(date)"
tail -n 30 temp/day12/day12e.log
