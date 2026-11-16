import json, os
R = os.environ["BFLC_ROOT"] + "/runs/"
def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict): out.update(flat(v, pre + k + "."))
        elif isinstance(v, (int, float)): out[pre + k] = v
    return out
for s in (0, 1):
    cards = {"B": flat(json.load(open(f"{R}e275/e273_p_s{s}.close1.5_t0.5/card.json"))), "c2": flat(json.load(open(f"{R}e292/e292_c2_s{s}.close1.5_t0.5/card.json"))), "a": flat(json.load(open(f"{R}e292/e292_a_s{s}.plain_t0.5/card.json"))), "ah": flat(json.load(open(f"{R}e292/e292_a_s{s}.close1.5_t0.5/card.json")))}
    keys = [k for k in cards["B"] if k in ("p.nodes.node_P", "p.nodes.node_R", "p.nodes.pos_err_med", "p.graph.junction_P", "p.graph.junction_R", "p.graph.junction_F1", "p.graph.vert100_dp", "p.graph.topo_F1", "p.graph.lanes_found", "p.graph.F1@1.0") and k in cards["a"]]
    print(f"seed {s}")
    for k in keys[:30]:
        print(f"  {k:38s} " + " ".join(f"{n} {cards[n].get(k, float('nan')):9.4f}" for n in ("B", "c2", "a", "ah")))
