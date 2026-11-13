#!/bin/bash
cd "$(dirname "$0")/../.."
for s in 0 1; do
  for k in thr0.1 snap_thr0.1 snap_nogeom_thr0.1 snap_all_thr0.1; do
    f=runs/e312_p_s$s/ends_card_$k.json
    if [ -f $f ]; then python3 - "$f" "s$s $k" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); n = c["numbers"]
h, b = n["head"], n["base"]
print(sys.argv[2], "topo", round(h["topo_F1"], 4), "(+%.4f)" % (h["topo_F1"] - b["topo_F1"]), "P", round(h["topo_P"], 4), "junction F1", round(h["junction_F1"], 4), "R", round(h["junction_R"], 4), "lanes", round(h["lanes_found"], 4), "F1@1", round(h["F1@1.0"], 4), "dup", round(h["dup_frac"], 4), "vert", round(h["vert100_dp"], 2), "| plain topo", round(b["topo_F1"], 4), "junction F1", round(b["junction_F1"], 4))
PY
    fi
  done
done
