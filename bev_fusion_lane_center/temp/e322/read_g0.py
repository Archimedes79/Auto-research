"""E322: the numbers of the gate G0 read from the result files (runs/<run>/wiring_g0*.json), scratch."""
import json
for run, f in (("e312_p_s0", "wiring_g0.json"), ("e312_p_s1", "wiring_g0.json"), ("e320_a_s0", "wiring_g0_attach.json"), ("e320_a_s1", "wiring_g0_attach.json")):
    d = json.load(open(f"runs/{run}/{f}")); n = d["numbers"]; st = d["stat"]
    g = lambda v, k: n[v][k]
    print(run, "o - c topo %+.5f dup %+.5f junction %+.5f lanes %+.5f | o_drop - o topo %+.5f | second - c topo %+.5f dup %+.5f | c - r6 %+.5f c - r3 %+.5f c - r2 %+.5f" % (
        g("o", "topo_F1") - g("c", "topo_F1"), g("o", "dup_frac") - g("c", "dup_frac"), g("o", "junction_F1") - g("c", "junction_F1"), g("o", "lanes_found") - g("c", "lanes_found"),
        g("o_drop", "topo_F1") - g("o", "topo_F1"), g("second", "topo_F1") - g("c", "topo_F1"), g("second", "dup_frac") - g("c", "dup_frac"),
        g("c", "topo_F1") - g("r6", "topo_F1"), g("c", "topo_F1") - g("r3", "topo_F1"), g("c", "topo_F1") - g("r2", "topo_F1")))
    print("   stat: matched %.1f %% of ends, found %.1f %% of ends (%.1f %% of matched), same %.1f %% of both, multi %.1f %% of both, multi_same %.1f %% of multi, tie %.1f %% of found" % (
        100 * st["matched"] / st["ends"], 100 * st["found"] / st["ends"], 100 * st["found"] / st["matched"], 100 * st["same"] / st["both"], 100 * st["multi"] / st["both"], 100 * st["multi_same"] / st["multi"], 100 * st["tie"] / st["found"]))
