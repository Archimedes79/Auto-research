"""E324 verdict by hand (replay of the stored cards, the rule of the card section 1, written before the run): reads temp/_session_scratch_s292/e324_table.json (made by e324_table.py from the cards), prints the clauses, the kill, the pre-registered additive numbers and the H_add / H_sel reading."""
import json, os
T = json.load(open(os.environ["BFLC_ROOT"] + "/temp/_session_scratch_s292/e324_table.json"))
g = lambda s, n, k: T[f"{s}|{n}"][k]  # noqa: E731
P, C, U, L, LN = "s  PRIMARY attach+nogeom", "c  E312 snapped", "u' E312 nogeom", "l  E316 snapped", "l' E316 nogeom"
add = {0: dict(dup=0.1866, topo=0.6879, junc=0.2936, f1=0.8760), 1: dict(dup=0.1626, topo=0.6985, junc=0.3109, f1=0.8780)}
add_excess = {0: 0.0122, 1: 0.0040}
sel_max = {0: 0.006, 1: 0.002}
ok_all = True
for s in (0, 1):
    d = lambda k, ref: g(s, P, k) - g(s, ref, k)  # noqa: E731
    clauses = {
        "topo_F1 >= u' + 0.005": d("topo_F1", U) >= 0.005,
        "F1@1.0 loss vs c <= 0.01": -d("F1@1.0", C) <= 0.01,
        "dup_frac vs c <= +0.01": d("dup_frac", C) <= 0.01,
        "vert100_dp vs c <= +1.0": d("vert100_dp", C) <= 1.0,
        "junction_F1 loss vs c <= 0.03": -d("junction_F1", C) <= 0.03,
    }
    kill = g(s, P, "topo_F1") < g(s, U, "topo_F1")
    print(f"seed {s}: topo_F1 s - u' {d('topo_F1', U):+.4f}, s - c {d('topo_F1', C):+.4f}, s - l' {d('topo_F1', LN):+.4f}, s - l {d('topo_F1', L):+.4f}")
    print(f"        dup_frac s - c {d('dup_frac', C):+.4f}, s - u' {d('dup_frac', U):+.4f}, s - l' {d('dup_frac', LN):+.4f}, s - l {d('dup_frac', L):+.4f}; F1@1.0 s - c {d('F1@1.0', C):+.4f}; junction s - c {d('junction_F1', C):+.4f}; vert s - c {d('vert100_dp', C):+.2f}; lanes found s - c {d('lanes_found', C):+.4f}")
    print("        clauses:", {k: bool(v) for k, v in clauses.items()}, "| kill (primary below u'):", kill)
    ok_all = ok_all and all(clauses.values())
    a = add[s]
    print(f"        additive prediction vs actual: dup {a['dup']:.4f} -> {g(s, P, 'dup_frac'):.4f} ({g(s, P, 'dup_frac') - a['dup']:+.4f}), topo {a['topo']:.4f} -> {g(s, P, 'topo_F1'):.4f} ({g(s, P, 'topo_F1') - a['topo']:+.4f}), junction {a['junc']:.4f} -> {g(s, P, 'junction_F1'):.4f} ({g(s, P, 'junction_F1') - a['junc']:+.4f}), F1@1.0 {a['f1']:.4f} -> {g(s, P, 'F1@1.0'):.4f}")
    ex = d("dup_frac", LN)
    print(f"        excess dup(s) - dup(l') {ex:+.4f}: additive value {add_excess[s]:+.4f} (H_add window {add_excess[s] - 0.003:+.4f} .. {add_excess[s] + 0.003:+.4f}), H_sel at most {sel_max[s]:+.4f} -> "
          + ("H_add" if abs(ex - add_excess[s]) <= 0.003 else "H_sel" if ex <= sel_max[s] else "between / outside"))
    o_c = g(s, "s  own kp snapped", "topo_F1") - g(s, C, "topo_F1")
    print(f"        own-keypoint snapped: topo s - c {o_c:+.4f}, dup {g(s, 's  own kp snapped', 'dup_frac') - g(s, C, 'dup_frac'):+.4f}, junction {g(s, 's  own kp snapped', 'junction_F1') - g(s, C, 'junction_F1'):+.4f}, F1@1.0 {g(s, 's  own kp snapped', 'F1@1.0') - g(s, C, 'F1@1.0'):+.4f}, vert {g(s, 's  own kp snapped', 'vert100_dp') - g(s, C, 'vert100_dp'):+.2f}, s - l {g(s, 's  own kp snapped', 'topo_F1') - g(s, L, 'topo_F1'):+.4f}")
print("ALL clauses hold on both seeds:", ok_all)
