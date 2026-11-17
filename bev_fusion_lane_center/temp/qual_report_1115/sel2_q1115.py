import os, json, pickle
import numpy as np
R = os.environ["BFLC_ROOT"]
c324 = pickle.load(open(f"{R}/runs/e324_s_s0/ends_card_snap_thr0.1.counts.pkl", "rb"))
c312 = pickle.load(open(f"{R}/runs/e312_p_s0/ends_card_snap_thr0.1.counts.pkl", "rb"))
rows = json.load(open(f"{R}/temp/qual_report_1115/perframe_table.json"))
def g(c): return c["rec"]["rs"]
print("false_len semantic: ", [ (k, g(c324['head'][222])[k]) for k in ('false_len','pred_len','n_p','n_g')])
cand = []
for k in range(873):
    h, b, e = c324["head"][k], c324["base"][k], c312["head"][k]
    cand.append(dict(k=k, row=rows[k]["row"], sc=rows[k]["scene"], falseH=g(h)["false_len"], predH=g(h)["pred_len"], falseB=g(b)["false_len"], falseE=g(e)["false_len"], dupH=h["topo"]["dup"][0] / 2, dupB=b["topo"]["dup"][0] / 2, dupE=e["topo"]["dup"][0] / 2, topoH=rows[k]["E324s_topo"], topoB=rows[k]["B_topo"], topoE=rows[k]["E312_topo"], ownP=rows[k]["E324s_ownP"], ownR=rows[k]["E324s_ownR"]))
print("-- highest false length E324s (m), with share")
for c in sorted(cand, key=lambda c: -c["falseH"])[:10]:
    print(f"k{c['k']} row{c['row']} sc{c['sc']} false H {c['falseH']:.1f} m of {c['predH']:.0f} m (B {c['falseB']:.1f}, E312 {c['falseE']:.1f}) topo {c['topoB']:.2f}/{c['topoE']:.2f}/{c['topoH']:.2f} ownR {c['ownR']:.2f} ownP {c['ownP']:.2f}")
print("-- highest duplicate length E324s (m)")
for c in sorted(cand, key=lambda c: -c["dupH"])[:10]:
    print(f"k{c['k']} row{c['row']} sc{c['sc']} dup H {c['dupH']:.1f} m of {c['predH']:.0f} (B {c['dupB']:.1f}, E312 {c['dupE']:.1f}) topo {c['topoB']:.2f}/{c['topoE']:.2f}/{c['topoH']:.2f}")
print("-- E324s dup minus E312 dup largest (E324s adds duplicates)")
for c in sorted(cand, key=lambda c: -(c["dupH"] - c["dupE"]))[:6]:
    print(f"k{c['k']} row{c['row']} sc{c['sc']} dup H {c['dupH']:.1f} E312 {c['dupE']:.1f} B {c['dupB']:.1f} topo {c['topoB']:.2f}/{c['topoE']:.2f}/{c['topoH']:.2f}")
tot = lambda key, cs: sum(g(c)[key] for c in cs)
print("pooled false_len/pred_len: B %.4f E312 %.4f E324s %.4f" % (tot('false_len', c324['base']) / tot('pred_len', c324['base']), tot('false_len', c312['head']) / tot('pred_len', c312['head']), tot('false_len', c324['head']) / tot('pred_len', c324['head'])))
