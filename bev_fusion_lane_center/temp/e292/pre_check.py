"""E292 PRECONDITION of the card (written before the run): c2 passes against c = topo F1 of c2 at least c - 0.01 on BOTH seeds (the guards are printed; the full precondition, with the assignment guards from the
qgraph_run_eval cards, is read by hand). exit 0 = precondition on topo F1 holds, 1 = it does not (the comparison would be VOID: the arm a is not run). Reads the cards of record and the tagged c2 cards of runs/e312_p_s<S>/."""
import json
import sys

ok = True
for s in (0, 1):
    d = f"runs/e312_p_s{s}/"
    c = json.load(open(d + "ends_card_snap_thr0.1.json"))["numbers"]["head"]
    c2 = json.load(open(d + "ends_card_c2_snap_thr0.1.json"))["numbers"]["head"]
    row = {k: round(c2[k] - c[k], 4) for k in ("topo_F1", "F1@1.0", "dup_frac", "vert100_dp", "junction_F1", "lanes_found")}
    print(f"seed {s}: c2 - c {row}")
    ok = ok and c2["topo_F1"] >= c["topo_F1"] - 0.01
print("PRECONDITION on topo F1:", "holds" if ok else "FAILS (comparison void, arm a not run)")
sys.exit(0 if ok else 1)
