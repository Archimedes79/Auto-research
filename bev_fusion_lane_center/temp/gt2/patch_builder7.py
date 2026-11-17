p = "temp/jr/build_junction_report.py"
s = open(p, encoding="utf-8").read()
old = '''        if un is None:
            c = J(f"runs/{d['run']}/ends_card.json")
            un = c if c and abs(c.get("thr", 0.5) - d["thr"]) < 1e-9 else None'''
assert old in s
new = '''        if un is None:
            for nm in (f"ends_card_thr{d['thr']}.json", "ends_card.json"):                                         # the unsnapped card of this threshold (runs/<run>/, written by the scorer itself since 2026-11-12) or the card of record
                c = J(f"runs/{d['run']}/{nm}")
                if c and abs(c.get("thr", 0.5) - d["thr"]) < 1e-9 and not c.get("snap"):
                    un = c
                    break'''
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
