import json
import sys

import numpy as np

sys.path.insert(0, "../../tools/viz")
sys.path.insert(0, "../../training/grid")
sys.path.insert(0, "../../training/data/prep")
import qgraph_view as QV
import jgt as J
import junction_paint as JP
import qjunction as QJ

tot = dict(ent=0, ent_p=0, ext=0, ext_p=0, ent_stop3=0, ent_ped4=0)
for sid, (name, toks) in QV.val_scenes().items():
    for t in toks:
        jg = J.junction_gt(QV.load_gt(t))
        if jg is None:
            continue
        paint = JP.load(t)
        pe, px = QJ.painted_flags(jg, paint)
        tot["ent"] += len(pe); tot["ent_p"] += int(pe.sum()); tot["ext"] += len(px); tot["ext_p"] += int(px.sum())
        pe_s, _ = QJ.painted_flags(jg, dict(stop_line=paint["stop_line"], ped_crossing=[]))
        pe_c, _ = QJ.painted_flags(jg, dict(stop_line=[], ped_crossing=paint["ped_crossing"]))
        tot["ent_stop3"] += int(pe_s.sum()); tot["ent_ped4"] += int(pe_c.sum())
S = dict(tot, entries_painted_share=tot["ent_p"] / tot["ent"], exits_painted_share=tot["ext_p"] / tot["ext"], entries_with_stop_line_share=tot["ent_stop3"] / tot["ent"], entries_with_crossing_share=tot["ent_ped4"] / tot["ent"])
print(json.dumps(S, indent=1))
json.dump(S, open("jpainted_val.json", "w"), indent=1)
