import json, sys, math
sys.path.insert(0, "/mnt/c/Auto-research/autoresearch/tools")
import card as CARD
f = sys.argv[1]
d = json.load(open(f))
m = d["metrics"]
# graft a better map like better_of + ms_better would write: every key but counts / info
bt = {k: "higher" for k in m if not k.startswith("info/") and not k.rsplit("/",1)[-1] == "n" and not k.rsplit("/",1)[-1].startswith("n_")}
d["better"] = bt
bad = CARD.validate(d)
print(len(bad), "problems")
import collections
c = collections.Counter(b.split(" ")[0] + " " + (b.split(" ")[-1] if "metric" in b else b[:60]) for b in bad)
for b in bad[:8]: print("  ", b[:160])
print("nan metrics:", sum(1 for v in m.values() if isinstance(v, float) and v != v), "of", len(m))
print("non-numbers:", [k for k, v in m.items() if isinstance(v, bool) or not isinstance(v, (int, float))][:5])
