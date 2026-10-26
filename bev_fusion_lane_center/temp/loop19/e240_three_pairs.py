"""E240: pair 3 (E240.a3 / c3) was cancelled (machine shared with other chats); the card is read with the three finished pairs, labelled in the summary."""
import json, sys
p = sys.argv[1]
d = json.load(open(p, encoding="utf-8"))
for arm in ("a", "c"):
    d["arms"][arm]["names"] = [n for n in d["arms"][arm]["names"] if not n.endswith("_s3")]
    d["arms"][arm]["seeds"] = [s for s in d["arms"][arm]["seeds"] if s != 3]
    d["arms"][arm]["note"] += " | pair 3 cancelled at 02:33 (machine shared with other chats): read with three pairs"
json.dump(d, open(p, "w", encoding="utf-8"), indent=1)
