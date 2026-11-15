#!/bin/bash
# live phase gates (end of phase 4 / 5) against the phase-7 record (same keys, final model) for B, r, p
cd "$(dirname "$0")/../.."
python3 - <<'PY'
import json, re, os
def gates(run):
    out = {}
    for ph in (4, 5, 6, 7):
        f = f"runs/{run}_p{ph}/log.txt"
        if not os.path.exists(f):
            continue
        for l in open(f):
            if " GATE " in l:
                out[ph] = json.loads(l.split(" GATE ", 1)[1])
    return out
keys = ["junction_F1", "topo_F1", "F1_1", "lanes_found", "Q", "pieces_per_gt_piece", "id_switches", "vert100_dp", "link_prec", "link_rec"]
for run in ("e273_p_s0", "e273_p_s1", "e274_r_s0", "e274_r_s1", "e274_p_s0", "e274_p_s1"):
    g = gates(run)
    if 7 not in g:
        print(run, "no phase 7 record"); continue
    print("==", run)
    for ph in (4, 5):
        live, fin = g.get(ph, {}), g[7].get(f"p{ph}", {})
        print(f"  end of phase {ph}  live / final model:", "  ".join(f"{k} {live.get(k, float('nan')):.3f}/{fin.get(k, float('nan')):.3f}" for k in keys if k in live or k in fin))
PY
