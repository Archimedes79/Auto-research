"""E240: pair 3 finished in loop 20 (E240.a3 / c3, pinned to the stored distance map): the card is read with four pairs again (names e24_j1{rc,eq}_s3 back into the arms)."""
import json
import sys

p = sys.argv[1]
d = json.load(open(p, encoding="utf-8"))
for arm, pre in (("a", "e24_j1rc_s"), ("c", "e24_j1eq_s")):
    a = d["arms"][arm]
    if pre + "3" not in a["names"]:
        a["names"].append(pre + "3")
    if 3 not in a["seeds"]:
        a["seeds"].append(3)
    a["note"] = a["note"].replace(" | pair 3 cancelled at 02:33 (machine shared with other chats): read with three pairs", " | pair 3 trained in loop 20 on the stored distance map (LCR_DIST_DIR), read with four pairs")
json.dump(d, open(p, "w", encoding="utf-8"), indent=1)
