"""Scratch (read-only): the stored cards that the E317 / E318 / E319 revision cites.  usage: python3 temp/e317/_read_cards.py"""
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def head(run, card):
    p = os.path.join(ROOT, "runs", run, card)
    if not os.path.exists(p):
        return None
    return json.load(open(p))["numbers"]["head"]


rows = [("unsnapped", "ends_card_thr0.1.json"), ("snap", "ends_card_snap_thr0.1.json"), ("snap-nogeom u'", "ends_card_snap_nogeom_thr0.1.json"), ("snap-all", "ends_card_snap_all_thr0.1.json")]
for S in (0, 1):
    for nm, f in rows:
        a = head(f"e312_p_s{S}", f)
        b = head(f"e320_a_s{S}", f)
        print(f"seed {S} {nm:16s} E312 {a['topo_F1'] if a else None}  E320 {b['topo_F1'] if b else None}  (dup {a['dup_frac'] if a else None} / {b['dup_frac'] if b else None})")
    for f in ("ends_card_snap_nogeom_attach_thr0.1.json", "ends_card_attach_thr0.1.json"):
        b = head(f"e320_a_s{S}", f)
        print(f"seed {S} E320 {f}: topo {b['topo_F1']:.4f} dup {b['dup_frac']:.4f} F1@1 {b['F1@1.0']:.4f} junction {b['junction_F1']:.4f} vert {b['vert100_dp']:.2f}")
    a = head(f"e312_p_s{S}", "ends_card_snap_thr0.1.json")
    b = head(f"e312_p_s{S}", "ends_card_thr0.1.json")
    e = head(f"e320_a_s{S}", "ends_card_thr0.1.json")
    es = head(f"e320_a_s{S}", "ends_card_snap_thr0.1.json")
    print(f"seed {S}: E320 own-keypoint unsnapped - E312 unsnapped = {e['topo_F1'] - b['topo_F1']:+.4f}; E320 own-keypoint snapped - E312 snapped = {es['topo_F1'] - a['topo_F1']:+.4f}")
    # E312 maps / pair numbers
    print("  E312 snap - unsnapped =", round(a["topo_F1"] - b["topo_F1"], 4))
    sa = head(f"e312_p_s{S}", "ends_card_snap_all_thr0.1.json")
    if sa:
        print("  E312 snap-all - unsnapped =", round(sa["topo_F1"] - b["topo_F1"], 4))
    for f in ("ends_card_snap_thr0.1.json", "ends_card_thr0.1.json"):
        for r in (f"e312_p_s{S}", f"e320_a_s{S}"):
            p = os.path.join(ROOT, "runs", r, f.replace(".json", ".counts.pkl"))
            print("  counts", r, f, os.path.exists(p))
