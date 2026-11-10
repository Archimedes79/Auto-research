import json
runs = [("B s0", "e273/e273_p_s0"), ("B s1", "e273/e273_p_s1"), ("r s0", "e274/e274_r_s0"), ("r s1", "e274/e274_r_s1"), ("p s0", "e274/e274_p_s0"), ("p s1", "e274/e274_p_s1")]
c = {n: json.load(open(f"runs/{p}/card.json")) for n, p in runs}
k0 = c["B s0"]
print("graph", list(k0["p"]["graph"])); print("nodes", list(k0["p"]["nodes"])); print("assign_exact", k0["p"]["assign_exact"]); print("diff", k0["diff"])
def row(label, f):
    print(f"{label:26s}", "  ".join(f"{n}:{f(c[n]):7.3f}" for n, _ in runs))
for key in ("topo_F1", "F1@1.0", "junction_F1", "lanes_found", "Q", "dup_frac", "vert100_dp", "topo_P", "topo_R"):
    try: row("graph " + key, lambda x, key=key: x["p"]["graph"][key])
    except KeyError: pass
for key in ("node_P", "node_R", "link_long_P", "link_long_R", "link_short_P", "link_short_R"):
    try: row("nodes " + key, lambda x, key=key: x["p"]["nodes"][key])
    except KeyError: pass
print()
for n, _ in runs:
    d = c[n]["diff"]; a = c[n]["p"]
    print(f"{n}: assignment p-g exact {d['exact']['mean']:+.4f} [{d['exact']['lo']:+.4f},{d['exact']['hi']:+.4f}]  map error 1-2 m {d['error']['mean']:+.4f} [{d['error']['lo']:+.4f},{d['error']['hi']:+.4f}]  lane_acc exact {a['assign_exact']['lane_acc']:.4f} error {a['assign_error']['lane_acc']:.4f}")
