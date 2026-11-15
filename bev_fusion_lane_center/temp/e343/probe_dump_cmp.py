import os, pickle, sys, json
R = os.environ["BFLC_ROOT"]
for sub in ("evaluation", "training", "training/grid", "training/data", ".."):
    sys.path.insert(0, os.path.join(R, sub))
import qgraph_eval as QE
card = json.load(open(f"{R}/temp/e343/val_max8/cards/card_s0.json"))
for name, f in (("snapped", "ends_card_snap_thr0.1"), ("snap_nogeom", "ends_card_snap_nogeom_thr0.1"), ("primary", "ends_card_snap_nogeom_attach_thr0.1"), ("stack", "ends_card_e338_snap_thr0.1")):
    st = pickle.load(open(f"{R}/runs/e324_s_s0/{f}.counts.pkl", "rb"))
    nh, nb = QE.graph_numbers(st["head"][:8]), QE.graph_numbers(st["base"][:8])
    c = card["variants"][name]["numbers"]
    print(f"{name:12s} head topo_F1 stored {nh['topo_F1']:.4f} fresh {c['head']['topo_F1']:.4f}   base stored {nb['topo_F1']:.4f} fresh {c['base']['topo_F1']:.4f}   lanes_found stored {nh['lanes_found']:.4f} fresh {c['head']['lanes_found']:.4f}")
