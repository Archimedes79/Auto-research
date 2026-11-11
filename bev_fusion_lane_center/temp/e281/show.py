"""E281 scratch: print the numbers of p12_s0.json / p12_s1.json that the card section 3a quotes"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
for s in (0, 1):
    r = json.load(open(os.path.join(ROOT, f"p12_s{s}.json")))
    print(f"=== seed {s}: frames {r['frames']} pairs {r['pairs']} convention {r['convention']['chosen']} {r['convention']['median_alignment_m']['pose acts on the lane frame']} m, signature {r['lane_signature']}, repro {r['repro_none_vs_E275_card']}")
    a = r["A"]
    print("A: delivered long", a["delivered_long"], "matched", a["matched"], "right", a["right"], "prec %.3f" % a["precision_matched"], "len<=11.4 %.3f" % a["share_len_le_11_4"], "persist %.3f n %d" % (a["persistence"], a["persistence_n"]),
          {k: (round(v["persist"], 3), v["n"]) for k, v in a["persistence_by_label"].items()})
    for lab in ("card_label", "alt_label"):
        b = r["B"][lab]
        print(f"B {lab}: n {b['n']} pos {b['pos']} band_n {b['band_n']} band_pos {b['band_pos']}")
        for k in ("S1", "S2", "S1+ (strong current)", "S2+ (strong + memory)"):
            print("   %-24s pooled %s  band %s" % (k, [round(x, 3) for x in b[k]["pooled"]], [round(x, 3) for x in b[k]["band"]]))
        for k in b:
            if k.startswith("gain"):
                print("   %-60s pooled %s band %s" % (k, [round(x, 3) for x in b[k]["pooled"]], [round(x, 3) for x in b[k]["band"]]))
    print("C:")
    for v, c in r["C"].items():
        print("   %-28s topo %.3f junc %.3f F1@1 %.3f lanes %.3f persist %.3f (n %d) delivered_long %d matched %d right %d prec_pair %.3f" % (v, c["topo_F1"], c["junction_F1"], c["F1@1.0"], c["lanes_found"], c["long_persist"], c["persist_n"], c["delivered_long"], c["matched_long"], c["right_long"], c["long_precision_pair"]))
