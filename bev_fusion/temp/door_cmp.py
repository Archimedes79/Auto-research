"""throw-away: compare two evaluated run copies (temp/door_check/<a> vs <b>): metrics.json (minus paths), card, table"""
import json
import os
import sys

R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "door_check"))
for x, y in (a.split(":") for a in sys.argv[1:]):
    a, b = (os.path.join(R, w, "t16_full_s0", "eval") for w in (x, y))
    ma, mb = (json.load(open(os.path.join(d, "metrics.json"))) for d in (a, b))
    for m in (ma, mb):
        m.pop("run"), m.pop("code")
    ca, cb = (json.load(open(os.path.join(d, "card.json"))) for d in (a, b))
    for c in (ca, cb):
        c.pop("date"), c.pop("provenance")
    diff = [k for k in ma["metrics"] if json.dumps(ma["metrics"][k]) != json.dumps(mb["metrics"].get(k))]
    rel = max((abs(ma["metrics"][k] - mb["metrics"][k]) for k in diff if isinstance(ma["metrics"][k], float) and ma["metrics"][k] == ma["metrics"][k]), default=0)
    print(f"{x} vs {y}: metrics.json equal={json.dumps(ma, sort_keys=True) == json.dumps(mb, sort_keys=True)} ({len(ma['metrics'])} numbers, {len(diff)} differ, max abs {rel:.2e}); "
          f"card equal={json.dumps(ca, sort_keys=True) == json.dumps(cb, sort_keys=True)}; "
          f"table equal={open(os.path.join(a, 'table.txt')).read() == open(os.path.join(b, 'table.txt')).read()}")
