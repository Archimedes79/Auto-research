#!/bin/bash
CELL="0.3,0.2,0.6,off"
IFS=, read LT LTL KT G <<< "$CELL"
GATE=$([ "$G" = off ] && echo --no-none-gate)
echo "[$LT] [$LTL] [$KT] [$GATE]"
