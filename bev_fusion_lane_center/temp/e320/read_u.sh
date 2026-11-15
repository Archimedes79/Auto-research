#!/bin/bash
cd "$(dirname "$0")/../.."
for s in 0 1; do f=runs/e312_p_s$s/ends_card_thr0.1.json; [ -f $f ] && echo "s$s unsnapped p>=0.1: $(grep -o '"topo_F1": *[0-9.]*' $f | sed -n 2p) (plain $(grep -o '"topo_F1": *[0-9.]*' $f | sed -n 1p))"; done
