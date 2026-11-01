#!/bin/bash
eval "$(sed -n "/^wincpu()/p" "$(dirname "$0")/run_e253.sh")"
echo "wincpu=$(wincpu)"
